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
    lag_interp = LargrangeInterpolation(x=x, y=y)
    lag_interp.fit_interp()
