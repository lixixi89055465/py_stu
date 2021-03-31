import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import os
from tensorflow.keras import datasets, layers, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)

print("X,Y range:", x.shape, y.shape)
print(x[:2])
print(y[:2])

X, Y = np.meshgrid(x, y)
print('X,Y maps :', x.shape, y.shape)

Z = himmelblau([X, Y])

print("Z.shape:")
print(Z.shape)

# fig = plt.figure("himmelblau")
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, Z)
# ax.view_init(60, -30)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# plt.show()

x = tf.constant([-3, 0.])
for step in range(200):
    with tf.GradientTape() as tape:
        tape.watch([x])
        y = himmelblau(x)
    grads = tape.gradient(y, [x])[0]
    x -= 0.01 * grads
    if step % 20 == 0:
        print("step {}: x= {}, f(x) = {}".format(step, x.numpy(), y.numpy()))
