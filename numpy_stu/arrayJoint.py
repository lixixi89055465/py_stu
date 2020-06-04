# coding=utf-8
import numpy as np

t1 = np.arange(0, 12).reshape(2, 6)
print(t1)
t2 = np.arange(12, 24).reshape(2, 6)
print(t2)
t3 = np.vstack((t1, t2))
print(t3)
t4 = np.hstack((t1, t2))
print(t4)
t1 = np.arange(12, 24).reshape(3, 4)
print(t1)
# 数组的行交换
t1[[1, 2], :] = t1[[2, 1], :]
print(t1)
# 数组的行交换
t1[:, [0, 2]] = t1[:, [2, 0]]
print(t1)

t = np.eye(4)
print((np.argmax(t, axis=0)))
# t[t == 1] = -1
t = np.argmin(t, axis=1)
print(t)

print(np.random.randint(10, 20, (4, 5)))