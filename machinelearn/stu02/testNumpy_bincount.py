# -*- coding: utf-8 -*-
# @Time    : 2023/9/1 11:05
# @Author  : nanji
# @Site    : 
# @File    : testNumpy_bincount.py
# @Software: PyCharm 
# @Comment :
import numpy as np

print(np.bincount(np.array([1, 1, 0, 0, 10])))
print('0'*100)
print(np.bincount(np.array([3])))

print('1'*100)
print(np.bincount(np.array([1, 2, 3, 4])))
