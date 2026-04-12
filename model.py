# -*- coding: utf-8 -*-
import torch  # PyTorch的核心基础库 它提供了PyTorch最底层的数据结构 即张量(Tensor)
# *Tensor可以把它理解为“能在显卡（GPU）上飞速运行的多维数组”  所有的图片、方向盘角度最终都要转换成Tensor才能被计算

import torch.nn as nn  # nn是Neural Network（神经网络）的缩写
# 代码里的卷积层（nn.Conv2d）全连接层（nn.Linear）激活函数（nn.ELU）防过拟合的Dropout（nn.Dropout）以及计算误差的损失函数（nn.MSELoss）全都来自这个库

import torch.optim as optim  # 优化器
# optim里的算法（比如optim.Adadelta）会根据误差一点点去修改模型里的参数（权重）让模型变得越来越聪明 控制学习快慢的（ReduceLROnPlateau）也在这个库

import pytorch_lightning as pl  # Lightning的主程序 我们简写为pl
# 代码里的pl.LightningModule（定义模型结构）和pl.Trainer（控制训练过程）都是这个库提供的 它会自动管理GPU、自动写好训练的for循环

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor  # callbacks叫做回调函数
# ModelCheckpoint在训练时监视验证集误差（val_loss）保存最佳模型权重（best_model.ckpt）
# EarlyStopping负责及时止损 如果它发现连续好几个Epoch模型的误差都没有再下降 它会直接终止训练
# LearningRateMonitor记录当前的“学习率”是多少 因为用了ReduceLROnPlateau（学习率会自动变小）这个函数可以让我们在日志里清楚地看到学习率的下降

from pytorch_lightning.loggers import CSVLogger  # 记录训练数据
# CSVLogger会把每一个Epoch的train_loss和val_loss写进一个Excel/CSV表格文件里 保存在logs/文件夹下

from torch.utils.data import DataLoader  # 数据加载器（传送带）
# 打包（Batching） 打乱（Shuffling） 多线程搬运（Multiprocessing）
# *多线程搬运（Multiprocessing）开启多个CPU核心（num_workers=4）同时去硬盘里读图片

# 导入配置和数据加载模块
from config import *  # *代表把config.py文件里定义的所有全局变量（比如CONFIG['batch_size'],CONFIG['input_height']等）全部导入
from load_data import split_train_val, DrivingDataset
# 从load_data.py文件里拿split_train_val这个函数 负责读取driving_log.csv 然后把数据切分成训练集和验证集
# 从load_data.py文件里拿DrivingDataset这个类 告诉PyTorch读取图片以及对单张图片做数据增强 DataLoader（传送带）就是从这个类里拿图片的


# 1.定义Lightning模型(把网络结构和训练逻辑写在一起) LightningModule是一个很重要的基类 它具有自动设备管理 自动训练循环 自动日志记录 自动保存机制的功能
class NvidiaLightningModel(pl.LightningModule):  # LightningModel：表示这个模型是用PyTorchLightning框架写的
    def __init__(self, input_channels=3, input_height=64, input_width=180):
        # 参数3,64,200：这是我们给机器人设定的默认眼睛（摄像头）规格 它默认接收RGB彩色图片（3通道） 图片高度64像素 宽度200像素
        super().__init__()  # 因为我们继承了pl.LightningModule
        # 在装我们自己的零件之前 先把父类（Lightning底盘）的初始化工作做完

        # 网络结构(和之前完全一样)
        self.conv_layers = nn.Sequential(  # nn.Sequential像流水线 数据从第一层进去 会自动一层一层往下传 不需要手动写传递过程
            nn.Conv2d(input_channels, 24, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),  # 卷积层用来提取图片的特征
            # 24是卷积核的数量 意味着经过这一层后的输出通道数 或者说提取的特征数量
            # kernel_size=(5, 5)：卷积核大小 stride=(2, 2)：步长 步长为2会让输出的图片长宽直接缩小一半 padding就是在图片外面补一圈0 保证边缘的信息不丢失
            nn.ELU(),  # 激活函数 给神经网络加入非线性
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),  # 最大池化层
            # 在一个2x2的小区域里 只保留数值最大的那一个（特征最明显的那一个）
            nn.Dropout(0.2),  # 每次训练随机丢弃20%的神经元 逼着剩下的神经元好好学习 防止模型过拟合

            nn.Conv2d(24, 36, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Dropout(0.2),

            nn.Conv2d(36, 48, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Dropout(0.2),

            nn.Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Dropout(0.2),

            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Dropout(0.2),
        )

        self.flatten_size = self._get_flatten_size((input_channels, input_height, input_width))
        # 全连接层必须提前知道卷积层最后会传过来多少个神经信号 调用_get_flatten_size这个函数来计算经过层层卷积、池化后图片里还剩下多少个数字

        self.fc_layers = nn.Sequential(  # 全连接层
            nn.Linear(self.flatten_size, 1152), nn.ELU(), nn.Dropout(0.5),  # nn.Linear(输入维度,输出维度)
            nn.Linear(1152, 100), nn.ELU(), nn.Dropout(0.5),  # 因为全连接层的参数量极其庞大 最容易发生过拟合 这里提高丢弃率
            nn.Linear(100, 50), nn.ELU(), nn.Dropout(0.5),
            nn.Linear(50, 10), nn.ELU(), nn.Dropout(0.5),
            nn.Linear(10, 1), nn.ELU()
        )
        # 关于为什么神经元数量最后是1 因为自动驾驶的回归任务 我们只需要模型输出一个数字（比如0.15代表右转 -0.2代表左转）

        self.criterion = nn.MSELoss()  # 损失函数 MSE均方误差

    def _get_flatten_size(self, shape):  # 在PyTorch里 全连接层（nn.Linear）必须明确知道输入进来的数据有多少个数字
        x = torch.zeros(1, *shape)  # torch.zeros是造一个全都是0的假数据 *shape就是把我们之前设定的(3,66,200)
        x = self.conv_layers(x)  # 让这张假图片在卷积层（流水线）里跑一遍
        return x.numel()  # numel()的意思是元素总数 即计算出的结果

    def forward(self, x):  # 这是模型真正工作时的流水线 规定了数据x（图片）进来后按什么顺序走
        # 图片的像素值是0~255下面这行代码把它们压缩到了-1.0到1.0之间 神经网络最喜欢这种小范围的数字 学得最快 如果是大数字会发生梯度爆炸
        x = x / 127.5 - 1.0  # 归一化
        x = self.conv_layers(x)  # 提取特征 把处理好的图片x送进卷积函数中处理
        x = x.view(x.size(0), -1)  # 这是PyTorch里的Flatten操作 把立体的特征图“拍扁”成一维数组 喂给后面的全连接层
        # x.size(0)代表这一批进来了多少张图片 我们必须保留这个维度
        # -1意思是除了第一维（128张图片）保持不变 剩下的维度自动乘在一起 拉平成一条直线 * -1是一个自动解方程的占位符
        x = self.fc_layers(x)  # 把展平的数据送进全连接层
        return x  # 经过全连接层得到的这个数字就是模型预测的方向盘转动角度

    # 告诉Lightning训练时干什么 *_batch_idx前面的下划线_ 意思是框架虽然把编号也传了但在这段代码里用不到它 只是占个位置
    def training_step(self, batch, _batch_idx):  # batch就是DataLoader传送带送过来的一批数据（128张图片）
        inputs, targets = batch  # inputs：128张路况图片 targets：128个真实的方向盘角度（目标值）
        outputs = self(inputs)  # 把图片喂给模型得到预测角度outputs 把图片inputs直接给模型self时会自动去调用我们的forward函数 outputs是128个模型预测的方向盘角度
        loss = self.criterion(outputs, targets)  # 由模型的outputs和真实角度targets计算均方误差

        # 自动记录loss到进度条 on_step=False//on_epoch=True意思就是不采用每一个batch记录一次误差而是采用每一个epoch记录一次平均误差
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)  # prog_bar=True会让loss实时显示端进度条上
        return loss  # 把loss return给Lightning框架 之后调用优化器去调整权重

    # 告诉Lightning验证时干什么
    def validation_step(self, batch, _batch_idx):  # 在Lightning框架里 一旦进入validation_step它会在后台自动关闭梯度计算以及随机丢弃 使之不会影响模型
        inputs, targets = batch  # Lightning会自动把数据源切换到验证集
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss  # val_loss非常重要 关系到学习率调整 模型保存 早停机制等

    # 配置优化器和学习率衰减(对应Keras的compile和ReduceLROnPlateau)
    def configure_optimizers(self):
        optimizer = optim.Adadelta(self.parameters())  # Adadelta是驱动模型更新权重的引擎
        # self.parameters()是把模型里几百万个神经元的“旋钮”（权重和偏置）全部交给了优化器 Adadelta是一种“自适应”优化器

        # patience=2意思是如果连续2轮（Epoch）验证集误差（val_loss）都没有下降 说明模型遇到瓶颈了
        # factor=0.1意思是这时候就把学习率缩小到原来的1/10
        # mode='min'告诉ReduceLROnPlateau让误差（Loss）越小越好
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
        return {  # 字典返回 在Lightning里我们只需要在初始化时把它们打包成一个字典return能自动触发优化器和调整器
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # 监控val_loss来决定是否降低学习率
            },
        }
    # *特别解释一下学习率调整 lr_scheduler是学习率调整器工具箱 ReduceLROnPlateau是其中最经典的一种调整策略


# 2.训练主程序(几乎和Keras一模一样)
if __name__ == '__main__':
    # 1.准备数据
    train_data, val_data = split_train_val('data_3/driving_log.csv')  # 修改路径同理

    train_loader = DataLoader(DrivingDataset(train_data, augment_data=True),  # num_workers=4意思是开启4个后台线程去读取数据
                              batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4,  persistent_workers=True)
    val_loader = DataLoader(DrivingDataset(val_data, augment_data=False),  # augment_data=False 验证时不需要数据增强和打乱
                            batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, persistent_workers=True)

    # 2.初始化模型
    model = NvidiaLightningModel(  # 实例化
        input_channels=CONFIG['input_channels'],
        input_height=CONFIG['input_height'],
        input_width=CONFIG['input_width']
    )

    # 3.定义Callbacks回调函数(完美对应Keras代码里的四个Callback)
    checkpoint_callback = ModelCheckpoint(  # 自动保存模型 save_top_k=1保证它永远只保留val_loss最低的那一次模型
        dirpath='logs/', filename='best_model_3', monitor='val_loss', save_top_k=1, mode='min'
    )
    # 早停 如果连续10轮（patience=10）模型都没进步 说明已经学到极限了
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min')  # 及时止损
    lr_monitor = LearningRateMonitor(logging_interval='epoch')  # 记录学习率下降的过程
    csv_logger = CSVLogger(save_dir='logs/', name='history')  # 把所有的train_loss val_loss记录整理成表格

    # 4.初始化Trainer(对应Keras的fit_generator配置)
    trainer = pl.Trainer(
        max_epochs=50,
        limit_train_batches=1200,  # 对应Keras的steps_per_epoch=1200 我们规定每个Epoch只从传送带(DataLoader)上拿1200批数据当做一轮结束
        limit_val_batches=300,  # 对应Keras的validation_steps=300
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=csv_logger,
        accelerator='auto',  # 自动识别是用GPU还是CPU Mac电脑还会自动用MPS
        devices=1,  # 使用1个GPU
        log_every_n_steps=1
    )

    # 5.开始训练(对应Keras的Model.fit_generator)
    trainer.fit(model, train_loader, val_loader)  # 按下启动按钮 自动循环、自动算梯度、自动更新权重、自动保存模型
    # train_fit把所有复杂的苦力都封装到了底层 省去了写循环 搬运数据 梯度清零等许多麻烦 它会自动调用功能函数 自动完  成循环 总之非常全能