# -*- coding: utf-8 -*-
# @Time : 2024/1/12 23:23
# @Author : nanji
# @Site : https://blog.csdn.net/zfhsfdhdfajhsr/article/details/109922718
# @File : test01.py
# @Software: PyCharm 
# @Comment :
import numpy as np

print(np.zeros(5))
print('0' * 100)
print(np.zeros((5,), dtype=int))
print('1' * 100)
print(np.zeros((2, 1)))
s = (2, 2)
print('2' * 100)
print(np.zeros(s))

print('3' * 100)
a = np.zeros((2,), dtype=[('x', 'i4'), ('y', 'i4')])
print(a)

print('4' * 100)
print(np.full(shape=(2, 2), fill_value=4))

