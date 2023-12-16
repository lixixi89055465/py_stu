# -*- coding: utf-8 -*-
# @Time    : 2023/9/22 14:15
# @Author  : nanji
# @Site    : 
# @File    : 48_2.py
# @Software: PyCharm 
# @Comment :
import numpy as np
a=np.arange(0,9).reshape(3,3)
print(a)
print(a.max())
print(a.max(axis=0))
print(a.max(axis=1))
