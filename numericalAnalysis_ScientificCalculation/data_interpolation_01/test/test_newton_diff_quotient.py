# -*- coding: utf-8 -*-
# @Time    : 2023/9/2 22:17
# @Author  : nanji
# @Site    : 
# @File    : test_lagrange.py
# @Software: PyCharm 
# @Comment :
import numpy as np

from numericalAnalysis_ScientificCalculation.data_interpolation_01.Newton_difference_quotient \
    import Newton_difference_quotient

# x = np.linspace(0, 2 * np.pi, 5, endpoint=True)
x = np.linspace(0, 2 * np.pi, 5, endpoint=True)
y = np.sin(x)
x0 = np.array([np.pi / 2, 2.158, 5.58, 4.784])

ndq = Newton_difference_quotient(x=x, y=y)
ndq.fit_interp()
# ndq.__diff_quotient__()
print("牛顿差商插值差商表 ")
print(ndq.diff_quot)

print('拉格朗日多项式如下:')
print(ndq.polynomial)
print('拉格朗日插值多项式系数向量和对应阶次 :')
print(ndq.poly_cofficient)
print(ndq.coefficient_order)
y0 = ndq.cal_interp_x0(x0)
print('所求插值点的值:', y0,'精确值是：',np.sin(x0))
print('0' * 100)
ndq.plt_interpolation(x0)
