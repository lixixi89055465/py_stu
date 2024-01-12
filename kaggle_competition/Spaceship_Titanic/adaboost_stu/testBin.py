# -*- coding: utf-8 -*-
# @Time : 2024/1/11 21:15
# @Author : nanji
# @Site : 
# @File : testBin.py
# @Software: PyCharm 
# @Comment :https://www.bilibili.com/video/BV13h4y147GK/?p=2&spm_id_from=pageDriver&vd_source=50305204d8a1be81f31d861b12d4d5cf

import numpy as np
import pandas as pd

np.random.seed(11)
x1 = np.array([1.2, 2.9, 2.6, 3.3, 2.0, 2.5, 1.4, 2.1, 1.7, 3.0])
x2 = np.array([4.7, 5.5, 3.9, 6.2, 3.5, 4.5, 5.1, 2.7, 4.1, 3.8])
x3 = np.random.randint(0, 2, 10)
x4 = np.random.randint(0, 2, 10)
y = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 1])
data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'y': y})
print(data)


