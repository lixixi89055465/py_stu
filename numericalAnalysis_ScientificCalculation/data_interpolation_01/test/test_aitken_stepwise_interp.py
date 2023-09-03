# -*- coding: utf-8 -*-
# @Time    : 2023/9/2 22:17
# @Author  : nanji
# @Site    : 
# @File    : test_lagrange.py
# @Software: PyCharm 
# @Comment :
import numpy as np

from numericalAnalysis_ScientificCalculation.data_interpolation_01.aitken_stepwise_interpolation import \
    AitkenStepwiseInterpolation

# x = np.linspace(0, 2 * np.pi, 10, endpoint=True)
# y = np.sin(x)
# x0 = np.array([np.pi / 2, 2.158, 3.58, 4, 784])

x = np.linspace(0, 2 * np.pi, 10, endpoint=True)
y = 2 * np.exp(-x) * np.sin(x)
x0 = np.array([np.pi / 2, 2.158, 3.58, 4, 784])

# x = np.linspace(0, 24, 13, endpoint=True)
# y = np.array([12, 9, 9, 10, 18, 24, 28, 27, 25, 20, 18, 15, 13])
# x0 = np.array([1, 10.5, 13, 18.7, 22.3])

asi_interp = AitkenStepwiseInterpolation(x=x, y=y)
asi_interp.fit_interp()
print('艾特肯多项式如下:')
print(asi_interp.polynomial)
print('艾特肯插值多项式系数向量和对应阶次 :')
print(asi_interp.poly_cofficient)
print(asi_interp.coefficient_order)
y0 = asi_interp.cal_interp_x0(x0)
print('所求插值点的值:', y0, '插值点的精度:', np.sin(x0))
print('0' * 100)
asi_interp.plt_interpolation(x0)
