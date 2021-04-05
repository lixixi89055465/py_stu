import tensorflow as tf
import os
from tensorflow.keras import datasets, layers, optimizers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)

x = tf.random.normal([1, 784])
w = tf.random.normal([784, 2])
b = tf.zeros([2])

logits = x @ w + b
print(logits.shape)
print(logits)
prob = tf.math.softmax(logits, axis=1)
print(prob.shape)
a = tf.losses.categorical_crossentropy([0, 1], logits, from_logits=True)
print(a)
print("prob:")
print(prob)
a = tf.losses.categorical_crossentropy([0, 1], prob)
print(a)
