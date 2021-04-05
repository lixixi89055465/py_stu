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
    logits = x @ w + b
    loss = tf.reduce_mean(tf.losses.categorical_crossentropy(tf.one_hot(y, depth=3), logits, from_logits=True))

grads = tape.gradient(loss, [w, b])

print("grads[0]:")
print(grads[0])
print("grads[1]:")
print(grads[1])
