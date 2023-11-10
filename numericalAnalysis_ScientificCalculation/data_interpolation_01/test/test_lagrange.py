# -*- coding: utf-8 -*-
# @Time    : 2023/9/2 22:17
# @Author  : nanji
# @Site    : 
# @File    : test_lagrange.py
# @Software: PyCharm 
# @Comment :
import numpy as np

from numericalAnalysis_ScientificCalculation.data_interpolation_01.largrange_interpolation import LargrangeInterpolation

x = np.linspace(0, 24, 13, endpoint=True)
y = np.array([12, 9, 9, 10, 18, 24, 28, 27, 25, 20, 18, 15, 13])
x0 = np.array([1, 10.5, 13, 18.7, 22.3])

lag_interp = LargrangeInterpolation(x=x, y=y)
lag_interp.fit_interp()
print('拉格朗日多项式如下:')
print(lag_interp.polynomial)
print('拉格朗日插值多项式系数向量和对应阶次 :')
print(lag_interp.poly_cofficient)
print(lag_interp.coefficient_order)
y0 = lag_interp.cal_interp_x0(x0)
print('所求插值点的值:', y0)
print('0' * 100)
lag_interp.plt_interpolation(x0)
