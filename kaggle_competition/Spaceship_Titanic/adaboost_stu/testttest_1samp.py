# -*- coding: utf-8 -*-
# @Time : 2024/1/12 23:36
# @Author : nanji
# @Site : 
# @File : testttest_1samp.py
# @Software: PyCharm 
# @Comment :
import numpy as np
from scipy  import stats
a = np.full(shape=(30, 355), fill_value=3)
b = np.random.normal(0, 3, 30)
print(a.shape)
print(b.shape)
factory_a = np.full(shape=(30, 355)) + np.random.normal(0, 3, 30)
factory_b = np.full(shape=(30, 355)) + np.random.normal(0, 3, 30)
# a_stat,a_pval=stats.ttest_1samp(a=factory_a,popmean=)

