# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/9/29 0:31
# @Author  : nanji
# @File    : testNP.digitize.py
# @Description :
import numpy as np

bins = np.array(range(-99, 102, 3))
print(bins)
a=np.digitize(-100,bins)# a=1
b=np.digitize(68,bins)# b=56
print(a)
print(b)
