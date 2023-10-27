# -*- coding: utf-8 -*-
# @Time    : 2023/10/27 14:15
# @Author  : nanji
# @Site    : 
# @File    : test06.py
# @Software: PyCharm 
# @Comment :
import numpy as np
a=np.random.randn(2,3)
b=np.mean(a,axis=0,keepdims=False)
print(b.shape)
