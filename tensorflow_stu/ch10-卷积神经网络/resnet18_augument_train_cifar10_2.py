import tensorflow as tf
import os

from Resnet18_34 import resnet18
from tensorflow.keras.callbacks import LearningRateScheduler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from Resnet18_34 import Resnet

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)

cpu_num = 16
config = tf.compat.v1.ConfigProto(device_count={"CPU": cpu_num},
                                  gpu_options=gpu_options,
                                  inter_op_parallelism_threads=cpu_num,
                                  intra_op_parallelism_threads=cpu_num,
                                  log_device_placement=True)
session = tf.compat.v1.Session(config=config)

import tensorflow as tf
import os
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
x_train=tf.convert_to_tensor(x_train)
x_test=tf.convert_to_tensor(x_test)
y_train=tf.squeeze(y_train, axis=1)
y_test=tf.squeeze(y_test, axis=1)

image_gen_train = ImageDataGenerator(#数据增强
                                     rescale=1,#归至0～1
                                     rotation_range=0,#随机0度旋转
                                     width_shift_range=0,#宽度偏移
                                     height_shift_range=0,#高度偏移
                                     horizontal_flip=True,#水平翻转
                                     zoom_range=1#将图像随机缩放到100％
                                     )
image_gen_train.fit(x_train)

#网络搭建 很重要！！
class vgg(Model):# 类 建网络
    def __init__(self):#
        super(vgg, self).__init__()
        self.c1=Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', input_shape=(32, 32, 3))
        self.b1=BatchNormalization()
        self.a1=Activation('relu')

        self.c2=Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', input_shape=(32, 32, 3))
        self.b2 =BatchNormalization()
        self.a2 =Activation('relu')

        self.p1=MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d1=Dropout(0.2)

        self.c3 =Conv2D(128, kernel_size=(3, 3), strides=1, padding='same')
        self.b3 =BatchNormalization()
        self.a3 =Activation('relu')
        self.c4 =Conv2D(128, kernel_size=(3, 3), strides=1, padding='same')
        self.b4 =BatchNormalization()
        self.a4 =Activation('relu')
        self.p2 =MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d2 =Dropout(0.2)

        self.c5 =Conv2D(256, kernel_size=(3, 3), strides=1, padding='same')
        self.b5 =BatchNormalization()
        self.a5 =Activation('relu')
        self.c6 =Conv2D(256, kernel_size=(3, 3), strides=1, padding='same')
        self.b6 =BatchNormalization()
        self.a6 =Activation('relu')
        self.c7 =Conv2D(256, kernel_size=(3, 3), padding='same')
        self.b7 =BatchNormalization()
        self.a7 =Activation('relu')
        self.ck =Conv2D(256, kernel_size=(3, 3), padding='same')
        self.bk =BatchNormalization()
        self.ak =Activation('relu')
        self.p3 =MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d3 =Dropout(0.2)

        self.c8 =Conv2D(512, kernel_size=(3, 3), strides=1, padding='same')
        self.b8 =BatchNormalization()
        self.a8 =Activation('relu')
        self.c9 =Conv2D(512, kernel_size=(3, 3), strides=1, padding='same')
        self.b9 =BatchNormalization()
        self.a9 =Activation('relu')
        self.c10 =Conv2D(512, kernel_size=(3, 3), padding='same')
        self.b10 =BatchNormalization()
        self.a10 =Activation('relu')
        self.cl =Conv2D(512, kernel_size=(3, 3), padding='same')
        self.bl =BatchNormalization()
        self.al =Activation('relu')
        self.p4 =MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d4 =Dropout(0.2)

        self.c11 =Conv2D(512, kernel_size=(3, 3), strides=1, padding='same')
        self.b11 =BatchNormalization()
        self.a11 =Activation('relu')
        self.c12 =Conv2D(512, kernel_size=(3, 3), strides=1, padding='same')
        self.b12 =BatchNormalization()
        self.a12 =Activation('relu')
        self.c13 =Conv2D(512, kernel_size=(3, 3), strides=1, padding='same')
        self.b13 =BatchNormalization()
        self.a13 =Activation('relu')
        self.cm =Conv2D(512, kernel_size=(3, 3), padding='same')
        self.bm =BatchNormalization()
        self.am =Activation('relu')
        self.p5 =MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d5 =Dropout(0.2)

        self.flatten =Flatten()
        self.f1 =Dense(512, activation='relu')
        self.d6 =Dropout(0.2)
        self.f2 =Dense(512, activation='relu')
        self.d7 =Dropout(0.2)
        self.f3 =Dense(10, activation='softmax')

    def call(self, x):#网络结构的表达
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)

        x = self.p1(x)
        x = self.d1(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)

        x = self.c4(x)
        x = self.b4(x)
        x = self.a4(x)

        x = self.p2(x)
        x = self.d2(x)

        x = self.c5(x)
        x = self.b5(x)
        x = self.a5(x)
        x = self.c6(x)
        x = self.b6(x)
        x = self.a6(x)
        x = self.c7(x)
        x = self.b7(x)
        x = self.a7(x)
        x = self.ck(x)
        x = self.bk(x)
        x = self.ak(x)

        x = self.p3(x)
        x = self.d3(x)

        x = self.c8(x)
        x = self.b8(x)
        x = self.a8(x)
        x = self.c9(x)
        x = self.b9(x)
        x = self.a9(x)
        x = self.c10(x)
        x = self.b10(x)
        x = self.a10(x)
        x = self.cl(x)
        x = self.bl(x)
        x = self.al(x)
        x = self.p4(x)
        x = self.d4(x)

        x = self.c11(x)
        x = self.b11(x)
        x = self.a11(x)
        x = self.c12(x)
        x = self.b12(x)
        x = self.a12(x)
        x = self.c13(x)
        x = self.b13(x)
        x = self.a13(x)
        x = self.cm(x)
        x = self.bm(x)
        x = self.am(x)
        x = self.p5(x)
        x = self.d5(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d6(x)
        x = self.f2(x)
        x = self.d7(x)
        y = self.f3(x)
        return y
model = vgg()
#sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])#配置好参数，程序在fit时才会使用运行
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])#配置好参数，程序在fit时才会使用运行

checkpoint_save_path = "./vgg192/mnist.ckpt"#断点存续
print(os.path)

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

'''
verbose：日志显示
verbose = 0 为不在标准输出流输出日志信息
verbose = 1 为输出进度条记录
verbose = 2 为每个epoch输出一行记录
注意： 默认为 1
'''

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 # monitor='loss',
                                                 save_best_only=True,
                                                 verbose=2)


#训练过程
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#               loss='sparse_categorical_crossentropy',
#               metrics=['sparse_categorical_accuracy'])
# history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=64), epochs=150,
#                     validation_data=(x_test, y_test),
#                     validation_freq=1, callbacks=[cp_callback], verbose=1,shuffle=True)
#
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
#               loss='sparse_categorical_crossentropy',
#               metrics=['sparse_categorical_accuracy'])
# history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=64), epochs=100,
#                     validation_data=(x_test, y_test),
#                     validation_freq=1, callbacks=[cp_callback], verbose=1,shuffle=True)

#测试是否达到这个准确率的实验过程
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=64), epochs=12,
                    validation_data=(x_test, y_test),
                    validation_freq=1, callbacks=[cp_callback], verbose=1,shuffle=True)

# model.evaluate(x=x_test,y=y_test)
model.summary()
