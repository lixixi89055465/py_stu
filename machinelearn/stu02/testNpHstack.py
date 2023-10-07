# -*- coding: utf-8 -*-
# @Time    : 2023/10/7 16:43
# @Author  : nanji
# @Site    : 
# @File    : testNpHstack.py
# @Software: PyCharm 
# @Comment :
import numpy as np
macro_avg=np.arange(6).reshape(2,3)
print('0'*100)
print(np.hstack(macro_avg))
print('1'*100)
print(np.hstack([macro_avg]))
