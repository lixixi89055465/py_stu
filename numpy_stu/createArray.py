# coding=utf-8
import numpy as np
import random

a = np.array([1, 2, 3, 4, 5, 6], dtype=np.int)
b = np.array(range(1, 6))
print(a, b)
print(type(a))
t2 = np.array(range(10))
print(type(t2))
t3 = np.arange(12)
print(t3)
t3 = np.arange(4, 11, 2)
print(type(t3))
print(t3.dtype)
t3 = np.array(range(1, 4), dtype=np.int8)
print(t3)
print(t3.dtype)
print(type(t3))
t3 = np.array([1, 1, 0, 1, 0, 0], dtype=bool)
print(t3)
print(type(t3))
print(t3.dtype)
# 调整数据类型
t6 = t3.astype("int8")
print("t6:", t6)
print(type(t6))
print(t6.dtype)

# numpy 中的小数
t7 = np.array([random.random() for i in range(10)])
print(t7)
print(t7.dtype)
t8 = np.round(t7, 2)
print("t8",t8)
print(t8.dtype)
