# -*- coding: utf-8 -*-
import csv  # 打开和读取driving_log.csv文件
import random  # 随机选相机（左/中/右）或 随机决定要不要翻转图片
import cv2  # OpenCV库 负责读取图片、裁剪、缩放、变色 实现翻转的是OpenCV库的cv2.flip()函数
import torch  # PyTorch核心库 负责把数字变成Tensor（张量）供模型学习
from torch.utils.data import Dataset, DataLoader  # 导入这个Dataset基类 它定义了PyTorch数据集的标准 DataLoader是打包传送数据的
from sklearn.model_selection import train_test_split  # 专门负责把数据集切割成“训练集”和“测试集”
from os.path import join  # 自动把文件夹名字和图片名字粘在一起
from config import *  # 导入定义的参数（比如图片高宽、修正值等）
import numpy as np  # 图片在电脑里就是一堆数字矩阵 运用numpy函数库来处理和运算
# 在此特地解释数据切割 如果模型只在训练集上跑（不切分）它可能会死记硬背那几条路 一旦换个弯道就撞车


def split_train_val(csv_driving_data, test_size=0.1):  # 原：test_size=0.2表示切出20%的数据作为“验证集”（用来考试）
    """
    逻辑不变：读取CSV并拆分
    """
    with open(csv_driving_data, 'r') as f:  # 1.打开CSV文件(f就像是一个打开的文件句柄)
        data_reader = csv.reader(f)  # 2.用csv库去读这个文件
        driving_data = [row for row in data_reader][1:]
        # 3.[row for row in data_reader][1:]的意思是把表格里的每一行都存进一个列表 但是[1:]表示跳过第一行（通常第一行是标题）

    # 4.暂时取消切割，把所有数据都给train_data，验证集给个空列表
    # train_data = driving_data
    # val_data = []
    # 原：4.调用train_test_split进行数据集切割 random_state=1保证每次切出来的结果都一样方便实验
    train_data, val_data = train_test_split(driving_data, test_size=test_size, random_state=1)
    return train_data, val_data  # 返回切好的两份数据列表


# 核心：定义数据读取类
class DrivingDataset(Dataset):  # self可以理解为这个类的钥匙
    def __init__(self, data_list, data_dir='data_3', augment_data=True, bias=CONFIG['bias']):  # 用哪个数据集就改成哪个图片路径
        self.data_dir = data_dir  # 保存图片地址
        self.augment_data = augment_data  # 模式开关 True训练训练 False验证模型
        # 根据bias过滤掉多余的直行数据
        # 1.准备一个空的新箱子
        filtered_list = []

        for row in data_list:  # 2.从旧箱子(data_list)里挑数据
            steer = float(row[3])
            # 如果方向盘角度（abs取了绝对值）接近0（直行）并且生成的随机数小于bias(具体见config)就跳过这条数据（相当于丢弃了80%的直行数据）
            if abs(steer) < 0.01 and random.random() < bias:
                continue
            # 否则就装进新箱子
            filtered_list.append(row)

        # 3.挑完之后把新箱子交给self.data_list
        self.data_list = filtered_list

    def __len__(self):
        # 告诉PyTorch数据总量有多少行
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        核心步骤：PyTorch会自动根据索引idx来调用这个函数 idx默认为0 1 2 3...
        """
    # 1.idx是行号 我们根据行号从CSV列表里拿出那一行图片信息 只处理当前这一张图
        row = self.data_list[idx]  # row[0]是中图 row[1]是左图 row[2]是右图 row[3]是角度
        # ct_path被赋值中间图片路径row[0] 依次类推赋值 对于"_"则是被赋一些不必要的值 类似于丢弃了
        ct_path, lt_path, rt_path, steer, _, _, _ = row
        steer = np.float32(steer)  # 调用np.float32把角度转成数字 方便计算
    # 2.随机选择摄像头（左、中、右）并进行角度补偿 生成地址img_path
        # 这里继承原有的逻辑
        camera = random.choice(['frontal', 'left', 'right'])  # random随机抽选一个视角
        if camera == 'frontal':
            img_path = join(self.data_dir, ct_path.strip())  # join函数可自动把csv里的路径和图片文件夹data拼起来
        elif camera == 'left':
            img_path = join(self.data_dir, lt_path.strip())  # self.data_dir是初始化时存进去的文件夹路径
            steer += CONFIG['delta_correction']  # 修正角度 左+
        else:  # right
            img_path = join(self.data_dir, rt_path.strip())  # strip()函数是去除字符串开头和结尾的“空白字符”（包括空格 换行符\n 制表符\t等）
            steer -= CONFIG['delta_correction']  # 右-修正角度

    # 3.读取并预处理
        image = cv2.imread(img_path)  # cv2.imread读进来的图片现在还是Numpy三维数组(H,W,C) 读入的颜色通道顺序是BGR
        if image is None:
            # 防错处理 万一某张图坏了返回下一张
            return self.__getitem__((idx + 1) % len(self.data_list))  # 递归调用getitem函数

        # 预处理 调用原有的预处理逻辑（裁剪、缩放）
        image = self.preprocess(image)  # 调用原来的预处理函数preprocess处理读入的numpy数组

    # 4. 数据增强（仅在训练集augment_data=True开启）
        if self.augment_data:
            image, steer = self.augment(image, steer)  # 调用augment函数进行数据增强

    # 5.维度转换：HWC->CHW
        # OpenCV是(高度,宽度,3) PyTorch要(3,高度,宽度) 3代表三个颜色通道RGB
        image = np.transpose(image, (2, 0, 1))

        # 6.torch.from_numpy函数把Numpy数组变成Torch的张量（Tensor）供模型计算
        return torch.from_numpy(image).float(), torch.tensor([steer], dtype=torch.float32)  # 返回：处理好的图片 处理好的转向角答案

    def preprocess(self, frame_bgr):  # 预处理函数
        """完全保留原有的裁剪和缩放逻辑"""
        h, w = CONFIG['input_height'], CONFIG['input_width']
        # 提取起点和终点
        top = CONFIG['crop_height'].start  # 60
        bottom = CONFIG['crop_height'].stop  # 140
        # 裁剪 只取图片中间高度部分  ":"分别表示宽度不变 颜色通道不变
        frame_cropped = frame_bgr[top:bottom, :, :]
        # 缩放
        frame_resized = cv2.resize(frame_cropped, dsize=(w, h))  # cv2.resize把大图片快速缩小数倍 dsize就是目标尺寸 (宽, 高)
        return frame_resized.astype('float32')  # 返回astype函数转numpy数组为浮点型的数组 供神经网络使用

    def augment(self, frame, steer):
        """完全保留原有的镜像和亮度调整逻辑"""
        # 随机翻转
        if random.random() > 0.5:  # random.random()生成0.0到1.0之间的随机数
            frame = cv2.flip(frame, 1)  # 翻转函数 frame是图片矩阵 1:表示水平翻转（左右颠倒）0:垂直翻转 -1:水平垂直都翻转
            steer *= -1.0  # 如果图片左右颠倒了 结果也要取反 如原本往左打的方向盘（负）就得变成往右打（正）

        # 角度噪声 在波动范围内random一个随机数 normal表示正态分布 用正态分布是因为逻辑上这种干扰不会产生大扰动
        steer += np.random.normal(loc=0, scale=CONFIG['augmentation_steer_sigma'])  # loc是中心点 scale是波动范围

        # 亮度调整 hsv是颜色、浓度、亮度
        if CONFIG['input_channels'] == 3:
            # 切换模式 bgr->hsv
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # cv2.cvtColor函数把图片从蓝绿红BGR色彩空间转换到色调饱和度亮度HSV空间 方便后面调整亮度
            # 范围内random一个倍数 亮度乘以倍数
            ratio = random.uniform(CONFIG['augmentation_value_min'], CONFIG['augmentation_value_max'])  # 随机调整亮度
            # 在NumPy数组里[:,:,2]代表选中所有的像素 但只取第三个通道（也就是V亮度）乘以ratio这个随机倍数
            hsv[:, :, 2] = hsv[:, :, 2] * ratio
            # 防止调整亮度后数字超过255（颜色最大值）
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)  # clip函数来控制范围
            # 调完亮度后必须把图片转回BGR格式 因为OpenCV的其他函数（比如显示图片、保存图片）和后续的预处理都习惯处理BGR
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # 返回新的图片和角度
        return frame, steer


if __name__ == '__main__':  # 保护锁 只有当直接运行这个文件时 才执行下列逻辑
    train_data, val_data = split_train_val('data_3/driving_log.csv')
    train_dataset = DrivingDataset(train_data, augment_data=True)

    train_loader = DataLoader(  # 打包图片数据 传送数据
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True  # shuffle洗牌 每跑完一轮（Epoch）会自动把那几万条数据打乱重排
    )