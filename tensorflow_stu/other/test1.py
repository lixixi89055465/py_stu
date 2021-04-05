import tensorflow as tf
import os
from tensorflow.keras import datasets, layers, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)

config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)

y = tf.random.normal([2, 3])
print(y)
print(y.shape)
print(tf.nn.softmax(y, axis=1))
a=tf.constant(4.0).numpy()
print(a)
