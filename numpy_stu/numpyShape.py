# coding=utf-8

import numpy as np

t1 = np.arange(12)
print(t1.shape)
t2 = np.array([[1, 2, 3], [4, 5, 6]])
print(t2)
print(t2.shape)

t3 = t2.reshape(3, 2)
print(t2)
print(t3)
t4 = np.arange(24).reshape(2, 3, 4)
print(t4)
print(t4.flatten())
t5 = t4
t5 = t5.reshape(4, 6)
t6 = np.arange(100, 124).reshape((4, 6))
print("t5", t5)
print("t6", t6)
print(t6 + t5)
t7 = t2.flatten()
print(t7)
print(t6 + t7)
# 扭转对角线
t2 = np.arange(24).reshape((4, 6))
print("t2:\n", t2)
t2 = t2.transpose()
print(t2)
print("t2.swapaxes(1, 0):")
print(t2.swapaxes(1,0))
