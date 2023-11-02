# -*- coding: utf-8 -*-
# @Time    : 2023/11/2 21:56
# @Author  : nanji
# @Site    : 
# @File    : Test04.py
# @Software: PyCharm 
# @Comment :
import numpy as np

a = np.array([[1, 2], [3, 4]])
ans = np.linalg.det(a)
print(ans)
b = np.array([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
ans = np.linalg.det(b)
print(ans)


print(6*(-2*7 - 5*8) - 1*(4*7 - 5*2) + 1*(4*8 - -2*2))
