End-to-End Autonomous Driving Simulation Platform based on Tuanjie Engine

## Project Overview
This project is a deep learning-based control system for autonomous driving. By collecting driving data (camera images and steering angles) from a simulation environment, we train an end-to-end neural network to achieve autonomous navigation in virtual scenarios.

## Key Features
Simulation-Driven: Deep integration with Tuanjie Engine (Unity China) virtual environments.

Flexible Architecture: The model architecture is migrated to the PyTorch Lightning framework, supporting rapid iteration and multi-hardware scaling.

End-to-End Learning: Direct mapping from raw sensory input (pixels) to vehicle control commands (steering/throttle).

## File Structure
model.py: Core script for neural network architecture and training.

drive.py: Server-side script for communication with the simulator. It processes real-time images and returns predicted steering angles and throttle.

load_data.py: Handles dataset loading, preprocessing, and data augmentation.

config.py: Global configuration and hyperparameter management.

test.py: Script for model testing and validation.

requirements.txt: List of required Python dependencies.

keras_backup.py: (Legacy) Backup of the original Keras-based training model (Note: Primarily for reference; the current environment uses PyTorch).

Note on Storage: To keep the repository lightweight, directories such as data/ (datasets), logs/ (training logs), and venv/ (virtual environments), as well as model weights (*.pth, *.h5, *.ckpt), are excluded via .gitignore.

## Installation & Setup
1. Environment Prerequisites
Python Version: 3.9.13 (Compatible with any 3.9.x version).

Virtual Environment: It is highly recommended to use a virtual environment rather than a global system installation.

2. Setting up with PyCharm
In the bottom-right corner of PyCharm, click on the Python Interpreter (it may show <No interpreter>).

Select Add New Interpreter -> Add Local Interpreter....

In the left sidebar, choose Virtualenv Environment.

Ensure New environment is selected.

Location: The default venv folder in your project directory.

Base interpreter: Select the path to your Python 3.9 executable.

Click OK and wait for PyCharm to initialize the environment.

3. Installing Dependencies
Open the Terminal tab at the bottom of PyCharm.

Ensure the prompt is prefixed with (venv), indicating the virtual environment is active.

Run the following command to install all necessary packages:

Bash
`pip install -r requirements.txt`
If the download speed is slow, you can use a mirror (e.g., Tsinghua Open Source Mirror):

Bash
`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

@ZhouChang (Lead): Model architecture design, repository management, and project coordination.