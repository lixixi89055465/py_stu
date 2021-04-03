import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


batchsz = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(60000).batch(batchsz)
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batchsz)


class MyDense(layers.Layer):
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_weight('w', [inp_dim, outp_dim])
        self.bias = self.add_weight('b', [outp_dim])

    def call(self, inputs, training=None):
        out = inputs @ self.kernel + self.bias
        return out


class MyModule(tf.keras.Model):
    def __init__(self):
        super(MyModule, self).__init__()
        self.f1 = MyDense(28 * 28, 256)
        self.f2 = MyDense(256, 128)
        self.f3 = MyDense(128, 64)
        self.f4 = MyDense(64, 32)
        self.f5 = MyDense(32, 10)

    def call(self, inputs, training=None):
        x = self.f1(inputs)
        x = tf.nn.relu(x)
        x = self.f2(x)
        x = tf.nn.relu(x)
        x = self.f3(x)
        x = tf.nn.relu(x)
        x = self.f4(x)
        x = tf.nn.relu(x)
        x = self.f5(x)
        return x


network = MyModule()
network.compile(optimizer=optimizers.Adam(lr=0.01),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

network.fit(db, epochs=5, validation_data=ds_val, validation_freq=2)
network.evaluate(ds_val)

network.save_weights('weight.ckpt')
print('save weights.')

del network

network = Sequential([layers.Dense(256, activation='relu'),
                      layers.Dense(128, activation='relu'),
                      layers.Dense(64, activation='relu'),
                      layers.Dense(32, activation='relu'),
                      layers.Dense(10)])
network.compile(optimizer=optimizers.Adam(lr=0.01),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.load_weights('weight.ckpt')
print('loaded weights')
network.evaluate(ds_val)
