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

x = tf.random.normal([1, 14, 14, 4])
pool = layers.MaxPool2D(2, strides=2)
out = pool(x)
print(out.shape)

pool = layers.MaxPool2D(3, strides=2)
out = pool(x)
print(out.shape)
out = tf.nn.max_pool2d(x, 2, strides=2, padding='VALID')
print(out.shape)
