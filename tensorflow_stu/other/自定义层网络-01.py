import tensorflow as tf
import os

# from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MyDense(tf.keras.layer):
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_variable('w', [inp_dim, outp_dim])
        self.bias = self.add_variable('b', [outp_dim])

    def call(self, inputs, training=None):
        out = inputs @ self.kernel + self.bias
        return out


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = MyDense(28 * 28, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = tf.nn.relu(x)
        x = self.fc2(inputs)
        x = tf.nn.relu(x)
        x = self.fc3(inputs)
        x = tf.nn.relu(x)
        x = self.fc4(inputs)
        x = tf.nn.relu(x)
        x = self.fc5(inputs)
        x = tf.nn.relu(x)
        return x

