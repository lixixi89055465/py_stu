# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/8/19 20:41
# @Author  : nanji
# @File    : test_largange.py
# @Description :  机器学习 https://space.bilibili.com/512662380

from numericalAnalysis_ScientificCalculation.data_interpolation_01.largrange_interpolation import LargrangeInterpolation
import numpy as np

if __name__ == '__main__':
    x = np.linspace(0, 2 * np.pi, 10, endpoint=True)
    y = np.sin(x)
    x0 = np.array([np.pi / 2, 2.158, 3.58, 4.784])
    lag_interp = LargrangeInterpolation(x=x, y=y)
    lag_interp.fit_interp()
    print('拉格朗日多项式如下:')
    print(lag_interp.polynomial)
    print('拉格朗日插值多项式系数向量和对应阶次:')
    print(lag_interp.poly_coefficient)
    print(lag_interp.coefficient_order)
    y0 = lag_interp.cal_interp_x0(x0)
    print("所求插值点的值:", y0, "精确值是:", np.sin(x0))
    print('0' * 100)
    lag_interp.plt_interpolation(x0, y0)
