import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)

a = tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.25, 0.25, 0.25, 0.25])
print(a)
a = tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.01, 0.97, 0.01, 0.01])
print(a)

a=tf.losses.BinaryCrossentropy()([1],[0.1])
print(a)
a=tf.losses.binary_crossentropy([1],[0.1])
print(a)

