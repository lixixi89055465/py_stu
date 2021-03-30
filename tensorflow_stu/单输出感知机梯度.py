import tensorflow as tf
import os
from tensorflow.keras import datasets, layers, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)

config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)

x = tf.random.normal([1, 3])
w = tf.ones([3, 1])
b = tf.ones([1])

y = tf.constant([1])

with tf.GradientTape() as tape:
    tape.watch([w, b])
    logits = tf.sigmoid(x @ w + b)
    loss = tf.reduce_mean(tf.losses.MSE(y, logits))
grads = tape.gradient(loss, [w, b])
print(grads)
