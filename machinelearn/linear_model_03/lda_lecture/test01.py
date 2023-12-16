# -*- coding: utf-8 -*-
# @Time    : 2023/9/23 下午12:26
# @Author  : nanji
# @Site    : 
# @File    : 48_2.py
# @Software: PyCharm 
# @Comment :
import numpy as np
a=np.diag((1,2,3))
w,v=np.linalg.eig(a)
print(w)
print('0'*100)
print(v)

print('2'*100)
a = np.array([[1, -2], [1, 4]])
w,v=np.linalg.eig(a)
print(w)
print(v)
