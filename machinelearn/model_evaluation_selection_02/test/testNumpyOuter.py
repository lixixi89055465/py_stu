# -*- coding: utf-8 -*-
# @Time    : 2023/9/6 23:26
# @Author  : nanji
# @Site    : 
# @File    : testNumpyOuter.py
# @Software: PyCharm 
# @Comment :

import numpy as np
x1 = [1,2,3]
x2 = [4,5,6]
outer = np.outer(x1,x2)
print(outer)

print('0'*100)
x1 = [[1,2],[3,4]]
x2 = [[1,1],[1,1]]
outer = np.outer(x1,x2)
print(outer)