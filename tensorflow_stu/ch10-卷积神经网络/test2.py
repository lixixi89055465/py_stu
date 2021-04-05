import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, regularizers

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)

config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
cpu_num = 16
config = tf.compat.v1.ConfigProto(device_count={"CPU": cpu_num},
                                  gpu_options=gpu_options,
                                  inter_op_parallelism_threads=cpu_num,
                                  intra_op_parallelism_threads=cpu_num,
                                  log_device_placement=True)

x = tf.random.normal([128, 32, 32, 9])
layer = layers.Conv2D(7, kernel_size=5, strides=2, padding='same')
out = layer(x)
print("out.shape:")
print(out.shape)
print("layer.kernel.shape:")
print(layer.kernel.shape)
print("layer.bias.shape:")
print(layer.bias.shape)
print("*" * 20)
x = tf.random.normal([1, 32, 32, 3])
w = tf.random.normal([5, 5, 3, 4])
b = tf.zeros([4])
print("x.shape:")
print(x.shape)
out = tf.nn.conv2d(x, w, strides=1, padding='VALID')
print("out.shape:")
print(out.shape)
out = out + b
print("out.shape:")
print(out.shape)

print("*" * 20)
out = tf.nn.conv2d(x, w, strides=2, padding='VALID')
print("out.shape:")
print(out.shape)
