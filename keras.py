# -*- coding: utf-8 -*-

from keras.models import Sequential  # 导入顺序模型，用于线性堆叠神经网络层
from keras.layers import Conv2D, Flatten, Dense, Dropout, Lambda, MaxPooling2D
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau

from config import *
from load_data import *  # 导入数据加载模块（包含数据预处理函数）


def get_nvidia_model(summary=True):
    """
    使用NVIDIA卷积神经网络模型，NVIDIA架构对应的keras模型
    “自动驾驶汽车的端到端的学习。
    """
    model = Sequential()  # 创建顺序模型（层按顺序堆叠）
    # 标准化的输入
    model.add(Lambda(lambda x: x / 127.5 - 1.0,
                     input_shape=(CONFIG['input_height'], CONFIG['input_width'], CONFIG['input_channels'])))
    # input_shape指定输入图像的尺寸：高度、宽度、通道数（从config中读取）

    # 第一个卷积层，24个卷积核，大小5x5，卷积模式same,激活函数elu
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='elu', ))  # 卷积层
    # 池化层,池化核大小2x2,
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    # dropout随机丢弃五分之一的网络连接，防止过拟合
    model.add(Dropout(0.2))

    # 第二个卷积层，36个卷积核，大小5x5，卷积模式same,激活函数elu
    model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='elu', ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.2))

    # 第三个卷积层，48个卷积核，大小5x5，卷积模式same,激活函数elu
    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='elu', ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.2))

    # 第四个卷积层，64个卷积核，大小3x3，卷积模式same,激活函数elu
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu', ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.2))

    # 第五个卷积层，64个卷积核，大小3x3，卷积模式same,激活函数elu
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='elu', ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.2))

    # 全连接层,展开操作，
    model.add(Flatten())  # 把多维数据"压扁"成一维数据 全连接层需要1D输入

    # 添加隐藏层神经元的数量和激活函数
    model.add(Dense(1152, activation='elu'))  # 用ELU确保每个神经元都能学到有用特征 用Dropout确保网络不过分依赖某些特征
    model.add(Dropout(0.5))

    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='elu'))

    if summary:  # 输出层
        model.summary()
    return model


if __name__ == '__main__':
    # 将udacity csv数据拆分为培训和验证
    train_data, val_data = split_train_val(csv_driving_data='data/driving_log.csv')

    # 获取网络模型并编译它(默认Adam opt)
    Model = get_nvidia_model(summary=True)
    Model.compile(optimizer='Adadelta', loss='mse')  # 学习率优化器Adadelta和损失函数mse

    # 模型架构的json转储
    with open('logs/model.json', 'w') as f:  # 保存模型结构
        f.write(Model.to_json())

    # 定义回调函数以保存历史记录和权重
    # 应用Checkpoint时，应在每次训练中观察到改进时输出模型权重。
    checkpointer = ModelCheckpoint('logs/model.hdf5')  # 保存最佳模型权重
    logger = CSVLogger(filename='logs/history.csv')  # 记录训练历史
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)  # 验证损失10轮不改善则停止训练 monitor='val_loss': 监控验证集损失
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, mode='auto')  # 验证损失停滞时 降低学习率

    # 开始训练 使用数据生成器fit_generator进行训练
    Model.fit_generator(generator=generate_data_batch(train_data, augment_data=False, bias=CONFIG['bias']),
                        steps_per_epoch=1200,  # 训练：每个epoch用1200个batch
                        epochs=50,  # nb_epoch=50,总共训练50个epoch
                        validation_data=generate_data_batch(val_data, augment_data=False, bias=1.0),  # 验证生成器
                        validation_steps=300,  # 每个epoch用300个batch验证
                        callbacks=[reduce_lr, checkpointer, logger]  # 调用回归函数 调节学习率，保存模型，记录日志
                        )