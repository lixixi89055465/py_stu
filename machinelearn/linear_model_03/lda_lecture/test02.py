# -*- coding: utf-8 -*-
# @Time    : 2023/9/23 下午7:59
# @Author  : nanji
# @Site    : 
# @File    : test02.py
# @Software: PyCharm
# @Comment :
import numpy as np
eig_values=np.asarray([0,0,1,1,1])
print(eig_values)
print('0'*100)
idx = np.argwhere(eig_values != 0)
print(idx)
print('1'*100)
b=np.where(eig_values != 0)
print(b)
