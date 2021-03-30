import tensorflow as tf
import os
from tensorflow.keras import datasets, layers, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)

config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)

x = tf.random.normal([2, 4])
w = tf.random.normal([4, 3])
b = tf.zeros([3])
y = tf.constant([2, 0])

with tf.GradientTape() as tape:
    tape.watch([w, b])
    prob = tf.nn.softmax(x @ w + b, axis=1)
    loss = tf.reduce_mean(tf.losses.MSE(tf.one_hot(y, depth=3), prob))

grads=tape.gradient(loss,[w,b])
print(grads[0])
print(grads[1])

