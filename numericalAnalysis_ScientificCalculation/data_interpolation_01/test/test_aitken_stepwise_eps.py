# -*- coding: utf-8 -*-
# @Time    : 2023/9/2 22:17
# @Author  : nanji
# @Site    : 
# @File    : test_lagrange.py
# @Software: PyCharm 
# @Comment :
import numpy as np

from numericalAnalysis_ScientificCalculation.data_interpolation_01.aitken_stepwise_interp_eps import \
    AitkenStepWiseInterpolationWithEpsilon

if __name__ == '__main__':


    x = np.linspace(0, 2 * np.pi, 10, endpoint=True)
    y = np.sin(x)

    x0 = np.array([np.pi / 2, 2.158, 3.58, 4, 784])

    asi_interp = AitkenStepWiseInterpolationWithEpsilon(x=x, y=y,eps=1e-2)
    y0=asi_interp.fit_interp(x0)
    print('艾特肯带精度要求，所求插值点的值：',y0,'\n所求插值点的精度:',np.sin(x0))
    print('每个插值点递推次数为:',asi_interp.recurrence_num)
    asi_interp.plt_interpolation()
