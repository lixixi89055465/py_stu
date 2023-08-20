# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/8/19 15:13
# @Author  : nanji
# @File    : testasarray.py
# @Description :  机器学习 https://space.bilibili.com/512662380

import numpy as np

# 将list 转换为ndarray
a = [1, 2]
b = np.asarray(a)
print(b.tolist())
# 如果对象本身未darray,且不改变dtype,则不会copy之
a = np.array([1, 2])
print('0' * 100)
print(a)
print('1' * 100)
print(np.asarray(a) is a)

# 如果对象本身即为ndarray,且改变dtype,则还会copy之
a = np.array([1, 2], dtype=np.float32)
print('2' * 100)
print(np.asarray(a, dtype=np.float32) is a)
print('3'*100)
print(np.asarray(a, dtype=np.float64) is a)
