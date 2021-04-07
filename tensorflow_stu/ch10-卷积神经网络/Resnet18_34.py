import tensorflow as tf
import os
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, regularizers


class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1,weight_decay=5e-4):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same',kernel_regularizer=regularizers.l2(weight_decay))
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same',kernel_regularizer=regularizers.l2(weight_decay))
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
            self.downsample.add(layers.BatchNormalization())
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def call(self, inputs, training=None):
        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu1 = self.relu(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        residual = self.downsample(inputs)
        add = layers.add([bn2, residual])
        # out = self.relu(add)
        out = tf.nn.relu(add)
        return out

class Resnet(tf.keras.Model):
    def __init__(self, layer_dims, num_classes=100):  # [2,2,2,2]
        super(Resnet, self).__init__()
        self.stem = Sequential([
            layers.Conv2D(64, (3, 3), strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
        ])
        self.layer1 = self.build_resblock(64, layer_dims[0], stride=1)
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # [b,c]
        x = self.avgpool(x)
        # [b,100]
        x = self.fc(x)
        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks


def resnet18():
    return Resnet([2, 2, 2, 2])


def resnet34():
    return Resnet([3, 4, 6, 3])
