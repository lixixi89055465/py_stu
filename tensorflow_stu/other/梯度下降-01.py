import tensorflow as tf
import os
from tensorflow.keras import datasets, layers, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

w = tf.constant(1.)
x = tf.constant(2.)
y = x * w
with tf.GradientTape() as tape:
    tape.watch([w])
    y2 = x * w
grad1 = tape.gradient(y, [w])
print(grad1)
with tf.GradientTape() as tape:
    tape.watch([w])
    y2 = x * w
grad2 = tape.gradient(y2, [w])
print(grad2)
# 不可重复调用
# grad2 = tape.gradient(y2, [w])
# print(grad2)

with tf.GradientTape(persistent=True) as tape:
    tape.watch([w])
    y2 = x * w
# 可以重复调用
grad2 = tape.gradient(y2, [w])
print(grad2)

b = tf.constant(0.)
with tf.GradientTape() as t1:
    with tf.GradientTape() as t2:
        y = x * w + b
    dy_dw, dy_db = t2.gradient(y, [w, b])
d2y_dw2 = t1.gradient(dy_dw, w)

print(d2y_dw2)
