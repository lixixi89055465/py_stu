import tensorflow as tf
import numpy as np
import pickle as p
import os

from tensorflow.keras import models, optimizers, regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow import keras

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

weight_decay = 5e-4
batch_size = 128
learning_rate = 1e-2
dropout_rate = 0.5
epoch_num = 200


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb')as f:
        datadict = p.load(f, encoding='iso-8859-1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y


def load_CIFAR(Foldername):
    train_data = np.zeros([50000, 32, 32, 3], dtype=np.float32)
    train_label = np.zeros([50000, 10], dtype=np.float32)
    test_data = np.zeros([10000, 32, 32, 3], dtype=np.float32)
    test_label = np.zeros([10000, 10], dtype=np.float32)

    for sample in range(5):
        X, Y = load_CIFAR_batch(Foldername + "/data_batch_" + str(sample + 1))

        for i in range(3):
            train_data[10000 * sample:10000 * (sample + 1), :, :, i] = X[:, i, :, :]
        for i in range(10000):
            train_label[i + 10000 * sample][Y[i]] = 1

    X, Y = load_CIFAR_batch(Foldername + "/test_batch")
    for i in range(3):
        test_data[:, :, :, i] = X[:, i, :, :]
    for i in range(10000):
        test_label[i][Y[i]] = 1

    return train_data, train_label, test_data, test_label


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


def scheduler(epoch):
    if epoch < epoch_num * 0.4:
        return learning_rate
    if epoch < epoch_num * 0.8:
        return learning_rate * 0.1
    return learning_rate * 0.01


if __name__ == '__main__':
    # gpu config
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

    # load data
    # train_images, train_labels, test_images, test_labels = load_CIFAR(
    #    '/home/user/PycharmProjects/ImageClassification/VGG-16-19/VGG-16-TF2/cifar-10-batches-py')
    # 不需要转换为one-hot,labels已经是one-hot
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    # (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()
    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    print('train_images.shape', train_images.shape)  # (50000, 32, 32, 3)
    print('train_labels.shape', train_labels.shape)  # (50000, 10)
    print('test_images.shape', test_images.shape)  # (10000, 32, 32, 3)
    print('test_labels.shape', test_labels.shape)  # (10000, 10)

    # get model
    model = VGG16()
    # show
    model.summary()

    # train
    sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    change_lr = LearningRateScheduler(scheduler)

    # Tensorboard
    import time
    from tensorflow.keras.callbacks import TensorBoard

    # 设置格式化模型名称，以时间为标记
    model_name = "mode-{}".format(int(time.time()))
    # 设定存储位置，每个模型不一样的路径
    tensorboard = TensorBoard(log_dir='vgg16_from_tensorflow20/{}'.format(model_name))

    data_augmentation = False # 改为False，Ture还有bug
    if not data_augmentation:
        print('Not using data augmentation')
        # 不进行数据扩充
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.fit(train_images, train_labels,
                  batch_size=batch_size,
                  epochs=epoch_num,
                  callbacks=[change_lr],
                  validation_data=(test_images, test_labels))
    else:
        print('Using real-time data augmentation')  # 实时数据增强
        ## tensorflow2.0 数据增强
        train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                     rotation_range=10,
                                                                     width_shift_range=0.1,
                                                                     height_shift_range=0.1,
                                                                     shear_range=0.1,
                                                                     zoom_range=0.1,
                                                                     horizontal_flip=False,
                                                                     fill_mode='nearest')
        train_datagen.fit(train_images)
        # 测试集不需要进行数据处理
        test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)  # 不增强验证数据
        # test_datagen.fit(test_images)

        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.fit(train_datagen.flow(train_images, train_labels, batch_size=128),  # flows是实时
                  steps_per_epoch=train_images.shape[0] // batch_size,  # steps_per_epoch: 每个epoch所需的steps,不能少
                  epochs=200,
                  callbacks=[change_lr],
                  validation_data=test_datagen.flow(test_images, test_labels, batch_size=128),
                  # validation_data = (test_images, test_labels),
                  validation_steps=test_images.shape[0] // batch_size,  # 这个也是不能少
                  # callbacks=[tensorboard]
                  )