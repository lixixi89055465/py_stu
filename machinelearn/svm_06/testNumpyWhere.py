# -*- coding: utf-8 -*-
# @Time    : 2023/10/5 9:37
# @Author  : nanji
# @Site    : 
# @File    : testNumpyWhere.py
# @Software: PyCharm 
# @Comment :
import numpy as np

a = np.arange(10)
print(a)
c = np.arange(100, 110)
b = np.where(a > 4, c, a)
print('0' * 100)
print(b)
b = np.where(a > 4)
print('5'*100)
print(b)
print('1' * 100)
print(b[0])
print('2' * 100)
print(c)

print('3' * 100)
print(a > 4.)
