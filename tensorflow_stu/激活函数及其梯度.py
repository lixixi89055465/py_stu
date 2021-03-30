import tensorflow as tf
import os
from tensorflow.keras import datasets, layers, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)

config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)

a = tf.linspace(-10., 10., 10)
with tf.GradientTape() as tape:
    tape.watch(a)
    y = tf.sigmoid(a)

grads = tape.gradient(y, [a])

print("a:")
print(a)
print("y:")
print(y)
print("grads:")
print(grads)
print('*' * 20)
a = tf.linspace(-5., 5., 10)
a = tf.tanh(a)
print(a)
print('*' * 20)

a = tf.linspace(-1., 1., 10)
print(a)
print(tf.nn.relu(a))

print(tf.nn.leaky_relu(a))
