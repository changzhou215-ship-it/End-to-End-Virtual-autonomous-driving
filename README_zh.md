 [English](README.md)
 
**项目简介**  :laughing: 

本项目是一个基于深度学习的自动驾驶仿真控制系统。通过采集仿真环境中的驾驶数据（图像与转向角），训练端到端的神经网络模型，最终实现车辆在虚拟环境中的自主驾驶。


 **核心特性** :smiley: 

仿真驱动：深度集成 Tuanjie Engine 虚拟场景

架构灵活：模型部分已迁移至 PyTorch Lightning 框架，支持快速迭代和多硬件缩放


 **文件结构说明**  :grinning: 

model.py / keras.py: 神经网络模型的构建与训练脚本。（keras在当前的环境下用不了 我只是做了一个备份原keras框架的训练模型）

drive.py: 与自动驾驶仿真器通信的服务器端脚本，负责接收实时图像并返回预测的转向角和油门。

load_data.py: 数据集的加载、预处理与数据增强。

test.py: 测试脚本。

config.py: 项目全局参数与超参数配置文件。

requirements.txt: 项目运行所需的 Python 依赖库列表。

 _注意_ ：为了保持代码仓库的轻量化，data/（数据集）、logs/（训练日志）、venv/（虚拟环境）以及所有模型权重文件（*.pth, *.h5, *.ckpt）已被 .gitignore 忽略，不会上传到 Git 仓库。


 **环境安装** : :anguished: 

本项目指定的Python版本为 3.9.13（3.9.x 均可）。请千万不要用系统的全局环境，一定要建虚拟环境！

1、在PyCharm右下角找到 <No interpreter> 或当前显示的 Python 版本，点击它，选择 Add New Interpreter -> Add Local Interpreter...。

2、在弹出的窗口左侧选择Virtualenv Environment（虚拟环境）。

3、勾选 New environment。
Location: 默认会在项目目录下生成一个venv文件夹（保持默认即可）。
Base interpreter: 下拉选择你电脑上的 Python 3.9 路径。

4、点击 OK，等待 PyCharm 在底部跑完进度条创建环境。

 
**安装依赖包**  :frowning: 

1、点击 PyCharm 最底部的 Terminal（终端） 标签页。

2、确认终端前面带有 (venv) 字样（这代表你已经进入了虚拟环境）。
在终端中输入以下命令并回车，耐心等待安装完成（可能需要几分钟到十几分钟，取决于网速）：

`pip install -r requirements.txt`

( 提示：如果下载速度太慢报错，可以加上清华源加速：pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple)



@周畅：模型架构设计、仓库管理及整体调度

