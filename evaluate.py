import csv#处理csv文件
import cv2#图像的载入处理与输出
import os#提供了多种文件处理方式
import numpy as np#可进行数组和矩阵的运算
from keras.models import load_model
from sklearn.utils import shuffle#将序列随机进行排序



batch_size = 1
recompute = True
driving_data_fname = 'C:/P3/data'
driving_log_fname = 'driving_log.csv'
driving_img_fname = 'IMG/'
num_cameras = 3
steering_correction_value = .2
images_folder = os.path.join(driving_data_fname, driving_img_fname)

# pre-process driving log and add additional steering values for 
# off-center cameras
with open(os.path.join(driving_data_fname, driving_log_fname)) as csv_file:
    reader = csv.reader(csv_file)

    image_paths = []
    measurements = []

    for line in reader:#按列读取文件
        if line[0] == 'center':
            continue  # skip first line if header
        for i in range(num_cameras):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            cur_path = os.path.join(images_folder, filename)
            tmp_measurement = float(line[3])
            if 'left' in filename:
                tmp_measurement += steering_correction_value
            elif 'right' in filename:
                tmp_measurement -= steering_correction_value
            image_paths.append(cur_path)
            measurements.append(tmp_measurement)
    XY = list(zip(image_paths, measurements))

def generator(samples, batch_size=batch_size):#深度学习中用生成器读取所有的图片数据
    num_samples = len(samples)
    while True:
        for start in range(0, num_samples, batch_size):
            batch_samples = samples[start:start+batch_size]
            images = []
            measurements = []

            for sample in batch_samples:
                path_name = sample[0]
                measurement = sample[1]
                tmp_img = cv2.imread(path_name)
                tmp_img = tmp_img[..., ::-1]  # turn into rgb
                images.append(tmp_img)
                measurements.append(measurement)

            X_train = np.array(images)
            y_train = np.array(measurements)

            yield shuffle(X_train, y_train)

# evaluate
test_generator = generator(XY)

from keras.models import Model
from keras.layers import Input, Dense

inputs = Input(shape=(784,))
x = Dense(64, activation='relu')(input())
x = Dense(64, activation='relu')(x)
y = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=y)
model.load_weights('D:/P3/logs/model.h5')

model = load_model('D:/P3/logs/model.h5')

#print (next(test_generator))
loss = model.evaluate_generator(generator=test_generator,val_samples=len(XY))
#accuracy=model_evaluate(X_train,y_train)
print ('\ntest loss',loss)
#print('accuracy',accuracy)
