# -*- coding: utf-8 -*-
# @Time    : 2023/8/31 11:05
# @Author  : nanji
# @Site    : 
# @File    : testnp_newaxis.py
# @Software: PyCharm 
# @Comment :
import numpy as np

x = np.arange(4)

print(x)
print(x[np.newaxis, :])
print(x[:, np.newaxis])

print('0' * 100)
x = np.random.rand(2, 3)
print(x)
print('1' * 100)
diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
print(diff)
print('2' * 100)
print(x.shape)
print(x[:, np.newaxis, :].shape)
print(diff.shape)
