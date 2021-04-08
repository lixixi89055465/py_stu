import tensorflow as tf
import os

from Resnet18_34 import resnet18
from tensorflow.keras.callbacks import LearningRateScheduler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, regularizers
from Resnet18_34 import Resnet

from tensorflow.keras import models, optimizers, regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow import keras

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)

cpu_num = 16
config = tf.compat.v1.ConfigProto(device_count={"CPU": cpu_num},
                                  gpu_options=gpu_options,
                                  inter_op_parallelism_threads=cpu_num,
                                  intra_op_parallelism_threads=cpu_num,
                                  log_device_placement=True)
session = tf.compat.v1.Session(config=config)

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
x_train = tf.convert_to_tensor(x_train)
x_test = tf.convert_to_tensor(x_test)
y_train = tf.squeeze(y_train, axis=1)
y_test = tf.squeeze(y_test, axis=1)

image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(  # 数据增强
    rescale=1,  # 归至0～1
    rotation_range=0,  # 随机0度旋转
    width_shift_range=0,  # 宽度偏移
    height_shift_range=0,  # 高度偏移
    horizontal_flip=True,  # 水平翻转
    zoom_range=1  # 将图像随机缩放到100％
)
weight_decay = 5e-4
epochs = 200


def VGG16():
    model = models.Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3),
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

    model.add(Flatten())  # 2*2*512
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model


image_gen_train.fit(x_train)

import matplotlib.pyplot as plt
import pandas as pd

# model=resnet18()
model = VGG16()
model.summary()
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=64), epochs=epochs,
                    validation_data=(x_test, y_test),
                    validation_freq=1, verbose=1, shuffle=True)


def plot_learning_curves(history, label, epochs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_' + label] = history.history['val_' + label]
    pd.DataFrame(data).plot(figsize=(8, 5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.savefig(os.path.basename(__file__) + '_' + label + '.jpg')
    plt.show()


plot_learning_curves(history, 'accuracy', epochs, 0, 1)
plot_learning_curves(history, 'loss', epochs, 0, 10)
