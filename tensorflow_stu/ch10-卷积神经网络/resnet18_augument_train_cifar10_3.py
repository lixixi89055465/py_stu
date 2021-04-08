import tensorflow as tf
import os

from Resnet18_34 import resnet18
from tensorflow.keras.callbacks import LearningRateScheduler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, regularizers
from Resnet18_34 import Resnet
from Resnet18_34 import resnet34

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

model = Resnet([3, 4, 6, 3], num_classes=10)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
model.build(input_shape=(None, 32, 32, 3))
model.summary()
history = model.fit(x_train, y_train, batch_size=64, epochs=200,
                    validation_data=(x_test, y_test),
                    validation_freq=1, verbose=1, shuffle=True)

import matplotlib.pyplot as plt
import pandas as pd

epochs = 200


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
