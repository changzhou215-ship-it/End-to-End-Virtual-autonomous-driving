# -*- coding: utf-8 -*-
import argparse  # 这个库没有用到 我们已经把路径写死在代码里了 但它本质上是Python自带的命令行参数解析器 也不用加路径代码 后面要换路径还是要在terminal里面敲命令行
# socketio接收模拟器发来的一堆乱码字符串 最终把角度发回给模拟器使汽车转弯
import socketio  # 通信工具 它允许模拟器和Python之间无延迟 双向地发送数据
# 下面三个库把socketio的字符串进行处理还原成一张图片
import base64  # 解码器 模拟器发来的是Base64格式的字符串 这个库负责把字符串重新翻译成二进制的字节流
from io import BytesIO  # 虚拟内存盘 BytesIO可以把刚才内存里的“字节流”伪装成一个真实的文件让图片库直接读取
from PIL import Image  # Python经典的图像库 它接收BytesIO伪装好的文件 正式把它打开成一张我们可以看到、可以操作的RGB彩色图片
# 下面两个库把图片裁剪 缩小 翻转变成矩阵
import cv2  # OpenCV图像处理库 用这个库把图片里的天空和车头裁剪掉 并且把图片缩小到66x200 最后把RGB颜色转换成BGR
import numpy as np  # 矩阵计算器 除了转换数组外 这里它把图片的维度从(高度,宽度,通道)翻转成PyTorch的(通道,高度,宽度)

import eventlet.wsgi
# eventlet是Python里一个非常有名的高并发网络库 可以在底层同时处理成千上万个网络连接绝不卡顿
# 这里实际用到了两个 eventlet.listen和eventlet.wsgi 作用过程比较复杂 总而言之一切为了通信
import time  # 时间工具
from flask import Flask  # Flask是一个非常轻量级的Web框架 作用是搭建一个Web服务器让模拟器连接过来
import torch  # PyTorch核心库 转换张量tensor给训练模型最后输出角度

# 用于记录上一帧的方向盘角度，实现平滑转向
prev_steering_angle = 0.0  # 全局变量 初始时是0.0

# 导入配置和我们的模型类
from config import *  # 见model
from model import NvidiaLightningModel

sio = socketio.Server()  # 实例化通信服务器 数据传输
app = Flask(__name__)  # 实例化Flask应用 为通信提供web服务器
model = None  # 提前声明一个全局变量 等程序启动时会把训练好的权重(.ckpt)加载进来赋值给它 供后续预测
# 自动检测是否有GPU 没有就用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess_image(frame_bgr):  # 注释详见load_data 这里单独写进来只是为了防止循环导入和依赖冲突
    """
    把预处理逻辑直接写在这里，防止从load_data导入类方法报错
    """
    h, w = CONFIG['input_height'], CONFIG['input_width']
    # 提取起点和终点
    top = CONFIG['crop_height'].start  # 60 裁剪逻辑也变了一点
    bottom = CONFIG['crop_height'].stop  # 140
    # 裁剪：只取图片中间高度部分
    frame_cropped = frame_bgr[top:bottom, :, :]
    # 缩放
    frame_resized = cv2.resize(frame_cropped, dsize=(w, h))  # 注释掉的话这个传入的参数直接就是frame_bgr
    return frame_resized.astype('float32')


@sio.on('telemetry')  # 触发器
def telemetry(sid, data):  # Socket.IO这个通信框架规定 任何客户端发来的消息(图片)都必须带上自己的专属号码即sid

    imgString = data["image"]  # 接收一堆图片字符串
    image = Image.open(BytesIO(base64.b64decode(imgString)))  # 调用base64函数转换字符串为二进制字节流
                                                              # 调用BytesIO把二进制数据伪造成一个文件给Image来open

    # image.save(CNN_get_current_time(), 'png') 注释掉了因为太占内存 下面的time代码也没有用到

    # 颜色通道对齐 来自模拟器的帧是RGB格式的 转成BGR以匹配我们训练时cv2.imread的习惯
    image_array = cv2.cvtColor(np.asarray(image), code=cv2.COLOR_RGB2BGR)  # np.asarray把“图片”变成“数字矩阵”

# 1.执行预处理(裁剪 调整大小等)
    image_array = preprocess_image(image_array)

# 2.维度转换：HWC->CHW(高度,宽度,通道->通道,高度,宽度)
    image_array = np.transpose(image_array, (2, 0, 1))

# 3.转成张量Tensor unsqueeze(0)的作用就是在最前面添加Batch维度(变成1,3,66,200) 因为我们只有一张图且维度是3维
    image_tensor = torch.from_numpy(image_array).float().unsqueeze(0)

    # 把数据(tensor)送到GPU（或CPU）
    image_tensor = image_tensor.to(device)

    # 声明使用全局变量 在内存中拿到新的prev角度
    global prev_steering_angle  # 上一帧方向盘的角度

# 4.模型预测
    with torch.no_grad():  # 告诉PyTorch现在是预测 不需要算梯度 提升预测速度
        prediction = model(image_tensor)  # 拿到模型给出的最原始的预测角度（这个值可能非常突兀）模型给出来的结果是一个Tensor容器 比如tensor([0.15])
        raw_steering_angle = float(prediction.item())  # item()把Tensor里的数字提取出来 变成普通的float 比如0.15

    # 1、关键一步！参考历史惯性！方向盘平滑滤波(Low-pass Filter/低通滤波器) 过滤高频信号（突变的角度）
    # 新角度 = 30%的模型预测 + 70%的上次角度 (让转向变得极其平滑，像加上了一个阻尼器)
    steering_angle = 0.3 * raw_steering_angle + 0.7 * prev_steering_angle  # 一阶指数平滑滤波器
    prev_steering_angle = steering_angle  # 更新记录

    # 2、关键二步！非对称转向修正！
    # 如果是左转（角度小于0），强行削弱它的转向力度！
    if steering_angle < 0:
        # 乘以0.7 表示把左转力度削弱30%（如果还是撞左墙，就改成 0.6；如果转不过弯了，就改成 0.8）
        steering_angle = steering_angle * 0.7

    # 3、关键三步！线性动态油门！
    # 基础油门0.12，方向盘打得越大，减速越平滑，最低不低于0.05
    throttle = 0.12 - abs(steering_angle) * 0.15  # 方向盘角度abs(steering_angle)打得越大，减掉的油门就越多
    throttle = max(0.05, throttle)  # max()函数限制最小油门，防止停下来

    print(f"预测方向盘: {steering_angle:>7.4f} | 油门: {throttle:>7.4f}")
    send_control(steering_angle, throttle)  # 把经过优化的数据发送给模拟器执行


@sio.on('connect')  # 这是Socket.IO的“听” 接收消息 触发这个connect函数
def connect(sid, _environ):
    print("Simulator Connected! ", sid)  # 连接成功 打印一个sid随机乱码
    send_control(0, 0)  # 算是一个测试指令 踩一脚刹车然后方向盘回正


def send_control(steering_angle, throttle):
    sio.emit("steer", data={  # 这是Socket.IO的“说” 发送消息 "steer"是频道暗号（指转向）
    # 把算好的steering_angle和throttle打包成一个字典（匿名）传递给模拟器
    # 特别解释一下data 因为Socket.IO规定了所有要发送的实际内容必须放在data=这个参数后面 模拟器才能正确接收
        'steering_angle': steering_angle.__str__(),  # __str__把数字变成字符串发过去 直接传数字容易丢失精度或者报错
        'throttle': throttle.__str__()
    }, skip_sid=True)  # 代码用了skip_sid=True 因为只有一辆车在跑所以就不用核对sid了


def CNN_get_current_time():  # 注释掉了 但这个函数获取的时间可以精确到毫秒
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d-%H-%M-%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "CNN-%s-%03d.png" % (data_head, data_secs)
    return time_stamp


if __name__ == '__main__':
    # 1.指定模型权重路径
    checkpoint_path = 'logs/best_model_3.ckpt'  # 用什么模型写什么路径
    print(f'Loading PyTorch model from: {checkpoint_path}')

    # 2.从ckpt文件恢复模型权重
    model = NvidiaLightningModel.load_from_checkpoint(checkpoint_path)  # .load_from_checkpoint()读取训练参数填入模型

    # 3.切换到评估模式 会关闭Dropout 否则车会乱开  为什么？因为神经元因为Dropout的存在一直在随机掉线
    model.eval()

    # 4.移动到相应设备
    # 前面我们在telemetry函数里把图片数据送到了显卡（image_tensor.to(device)）模型必须和张量在同一个地方才能处理数据
    model.to(device)  # 把模型搬到了显存
    print("Model loaded successfully! Waiting for simulator to connect...")

    # 使用engineio的中间件Middleware打包sio(数据传输)和app(web服务器)
    app = socketio.Middleware(sio, app)

    # 部署为eventlet WSGI服务器
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)  # 4567是一个通信接口