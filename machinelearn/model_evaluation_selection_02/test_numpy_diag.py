# -*- coding: utf-8 -*-
# @Time    : 2023/9/1 22:51
# @Author  : nanji
# @Site    : 
# @File    : test_numpy_diag.py
# @Software: PyCharm 
# @Comment :

import numpy as np

import numpy as np

a = np.arange(1, 4)
b = np.arange(1, 10).reshape(3, 3)
print(b)

print('0'*100)
print(np.diag(b))
print('1'*100)
print(np.diag(a))
print('2'*100)