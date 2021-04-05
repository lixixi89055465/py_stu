import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)

# config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
cpu_num = 16
config = tf.compat.v1.ConfigProto(device_count={"CPU": cpu_num},
                                  gpu_options=gpu_options,
                                  inter_op_parallelism_threads=cpu_num,
                                  intra_op_parallelism_threads=cpu_num,
                                  log_device_placement=True)

session = tf.compat.v1.Session(config=config)


def preprocess(x, y):
    x = 2. * tf.cast(x, dtype=tf.float32) / 255. - 1.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batchsz = 128
# batchsz = 256
(x, y), (x_val, y_val) = datasets.cifar10.load_data()

y = tf.squeeze(y)
y = tf.one_hot(y, depth=10)
y_val = tf.squeeze(y_val)
y_val = tf.one_hot(y_val, depth=10)
print("x.shape:", x.shape)
print("x_val.shape:", x_val.shape)
print("y.shape:", y.shape)
print("y_val.shape:", y_val.shape)
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(50000).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.map(preprocess).batch(batchsz)

sample = next(iter(train_db))
print(sample[0].shape, sample[1].shape)


class MyDense(layers.Layer):
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_variable('w', [inp_dim, outp_dim])
        self.bias = self.add_variable('b', [outp_dim])

    def call(self, inputs, training=None):
        x = inputs @ self.kernel + self.bias
        return x


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = MyDense(32 * 32 * 3, 512)
        self.fc2 = MyDense(512, 512)
        self.fc3 = MyDense(512, 512)
        self.fc4 = MyDense(512, 512)
        self.fc5 = MyDense(512, 10)

    def call(self, inputs, training=None):
        x = tf.reshape(inputs, [-1, 32 * 32 * 3])
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)
        return x


network = MyModel()
network.compile(
    optimizer=optimizers.Adam(lr=1e-3),
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

network.fit(train_db, epochs=15, validation_data=test_db,
            validation_freq=1)
network.evaluate(test_db)
network.save_weights('ckpt/weights.ckpt')
print("save to ckpt /weights.ckpt")

network = MyModel()
network.compile(
    optimizer=optimizers.Adam(lr=1e-4),
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
network.load_weights('ckpt/weights.ckpt')
print("loaded weights from file.")
network.evaluate(test_db)
