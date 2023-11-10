# -*- coding: utf-8 -*-
# @Time    : 2023/9/25 下午9:28
# @Author  : nanji
# @Site    : 
# @File    : testPercentile.py
# @Software: PyCharm 
# @Comment :
import numpy as np

a=range(1,101)
print(a)
print(np.percentile(a, 90))

a=range(101,1,-1)
print(np.percentile(a, 90))

a = np.array([[10, 7, 4], [3, 2, 1]])
print(a)
print('1'*100)
print(np.percentile(a, 50, axis=0))

print('2'*100)
print(np.percentile(a, 30, axis=0,keepdims=False))
print('3'*100)
print(np.percentile(a, 30, axis=0,keepdims=True))
