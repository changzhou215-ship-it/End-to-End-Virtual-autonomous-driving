# -*- coding: utf-8 -*-

"""
Project 3 超参数配置

CONFIG = {
    'batch_size': 128,   # 512
    'input_width': 180,  # 320*240的图像剪裁的行高和列宽
    # 行高=240经过剪裁到120然后按比例压缩到66
    # 列宽=360不剪裁直接按比例压缩到200
    'input_height': 64,
    'input_channels': 3,  # 图片维度，灰度图像为1维，彩色图像为3维
    'delta_correction': 0.18,  # 每次转向偏置0.25
    'augmentation_steer_sigma': 0.10,  # 0.2
    'augmentation_value_min': 0.2,
    'augmentation_value_max': 1.5,
    'bias': 0.8,
    'crop_height': range(74, 140),
}
"""

CONFIG = {
    'batch_size': 128,   # 看完128张图调整1次参数 调整权重（什么重要）和神经网络阈值（多容易反应）
    'input_width': 180,  # 最终进入CNN神经网络的图片宽 高 颜色通道
    'input_height': 64,
    'input_channels': 3,
    'delta_correction': 0.08,  # 原版：每次转向偏置0.18 当抽中“左/右摄像头”时 方向盘强行加减的修正值
    'augmentation_steer_sigma': 0.10,  # 这是normal的集中程度 0.10表示方向盘随机抖动比较温和 正态分布集中
    'augmentation_value_min': 0.2,
    'augmentation_value_max': 1.5,  # 是hsv调节亮度的范围 最暗会变成原来的20% 最亮会增加50%
    'bias': 0.5,  # 筛选数据 随机过滤掉80%直行的数据
    'crop_height': range(60, 140),  # 裁切的图片高度范围(60, 140)
}