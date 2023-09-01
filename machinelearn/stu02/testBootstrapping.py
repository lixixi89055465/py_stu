# -*- coding: utf-8 -*-
# @Time    : 2023/9/1 15:33
# @Author  : nanji
# @Site    : 
# @File    : testBootstrapping.py
# @Software: PyCharm 
# @Comment :
import numpy as np


def bootstrapping(m):
    bootstrap = []
    for j in range(m):
        bootstrap.append(np.random.randint(0, m, 1))
    return np.array(bootstrap)


