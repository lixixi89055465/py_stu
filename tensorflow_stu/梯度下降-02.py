import tensorflow as tf
import os
from tensorflow.keras import datasets, layers, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

b = tf.constant(1.)
x = tf.constant(2.)
w = tf.constant(3.)

with tf.GradientTape() as t1:
    with tf.GradientTape() as t2:
        y = x * w + b
    dy_dw, dy_db = t2.gradient(y, [w, b])
d2y_dw2 = t1.gradient(dy_dw, w)

print(d2y_dw2)
