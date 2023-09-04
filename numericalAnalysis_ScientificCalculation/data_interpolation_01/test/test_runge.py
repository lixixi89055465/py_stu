# -*- coding: utf-8 -*-
# @Time    : 2023/9/3 10:25
# @Author  : nanji
# @Site    :  https://www.bilibili.com/video/BV1Lq4y1U7Hj/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=50305204d8a1be81f31d861b12d4d5cf
# @File    : test_runge.py
# @Software: PyCharm 
# @Comment :
import matplotlib.pyplot as plt
import numpy as np
from numericalAnalysis_ScientificCalculation.data_interpolation_01.largrange_interpolation import LargrangeInterpolation


def fun(x):
    '''
    龙格函数
    :param x: 标量，向量，矩阵
    :return:
    '''
    return 1 / (x ** 2 + 1)


if __name__ == '__main__':
    plt.figure(figsize=(8, 6))
    for n in range(3, 12, 2):
        x = np.linspace(-5, 5, n, endpoint=True)
        y = fun(x)
        lag_interp = LargrangeInterpolation(x, y)
        lag_interp.fit_interp()
        plt.plot(x, y, ls='-', color='b', label='train: %d' % (n - 1))
        # plt.plot(x, y+1, label='tes01')
        xi = np.linspace(-5, 5, 100, endpoint=True)
        yi = lag_interp.cal_interp_x0(xi)
        plt.plot(xi, yi, ls='-.', lw=0.7, color='r', label='test %d' % (n - 1))
    plt.plot(xi, fun(xi), 'k-', label=r'$\frac{1}{1+x^{2}}\qquad$')
    plt.xlabel('x-', fontdict={'fontsize': 12})
    plt.ylabel('y-', fontdict={'fontsize': 12})
    plt.title('Runge phenomenon')
    plt.legend()
    plt.show()
