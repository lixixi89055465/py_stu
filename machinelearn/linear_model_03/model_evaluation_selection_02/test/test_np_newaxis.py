# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/8/19 17:00
# @Author  : nanji
# @File    : test_np_newaxis.py
# @Description :  机器学习 https://space.bilibili.com/512662380

import numpy as np

x = np.arange(4)
print(x)
print(x[np.newaxis, :])
print('0' * 100)
print(x[:, np.newaxis])

x = np.random.rand(10, 2)
print('1' * 100)
print(x)
diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
print(x.shape)
print(x[:, np.newaxis, :].shape)
print('2' * 100)
print(diff.shape)
x=x[:,np.newaxis,:]
print('3'*100)
print(x.shape)
