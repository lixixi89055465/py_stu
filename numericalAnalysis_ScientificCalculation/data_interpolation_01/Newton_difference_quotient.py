# -*- coding: utf-8 -*-
# @Time    : 2023/10/15 下午8:23
# @Author  : nanji
# @Site    : 
# @File    : Newton_difference_quotient.py
# @Software: PyCharm 
# @Comment :
import numpy as np  # 数值运算，尤其是向量化的运算方式
import sympy  # 符号运算库
import matplotlib.pyplot as plt
from numericalAnalysis_ScientificCalculation.data_interpolation_01.utils import interp_utils
import pandas as pd


class Newton_difference_quotient:
    '''
    牛顿差商插值必要参数的初始化，及各健壮性的检测
    '''

    def __init__(self, x, y):
        '''
        牛顿差商插值必要参数的初始化，及各健壮性的检测
        @param x: 已知数据的x座标点
        @param y: 已知数据的y座标点
        '''
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)  # 类型转换,数据结构采用array
        if len(self.x) > 1 and len(self.x) == len(self.y):
            self.n = len(self.x)  # 已知离散数据点的个数
        else:
            raise ValueError("插值数据(x,y)维度不匹配!")
        self.polynomial = None  # 最终的插值多项式,符号表示
        self.poly_cofficient = None  # 最终插值多项式的系数向量，幂次从高到低
        self.coffient_order = None  # 对应多项式系数的阶次
        self.y0 = None  # 所求插值点的值，单个值或向量

    def __diff_quotient__(self):
        '''
        计算牛顿插值 (均差)
        :return:
        '''
        diff_quot = np.zeros((self.n, self.n))  # 存储差商
        diff_quot[:, 0] = self.y  # 差商表中第一列的值存储y值
        for j in range(1, self.n):  # 按列计算
            for i in range(j, self.n):  # 按列计算
                diff_quot[i, j] = \
                    (diff_quot[i, j - 1] - diff_quot[i - 1, j - 1]) / \
                    (self.x[i] - self.x[i - j])

        self.diff_quot = pd.DataFrame(diff_quot)
        return diff_quot

    def fit_interp(self):
        '''
        核心算法：生成牛顿差商插值多项式
        @return:
        '''
        diff_quot = self.__diff_quotient__()  # 计算差商
        d_q = np.diag(diff_quot)  # 构造牛顿差商插值，只需要对角线的值即可

        # 数值运算，符号运算
        t = sympy.Symbol("t")  # 定义符号变量
        self.polynomial = d_q[0]  # 插值多项式实例化
        term_poly = t - self.x[0]
        for i in range(1,self.n):
            # 针对每个数据点，构造插值基函数
            self.polynomial += d_q[i] * term_poly  # 插值多项式累加
            term_poly *= (t - self.x[i])
        # 插值多项式特征
        self.polynomial = sympy.expand(self.polynomial)
        polynomial = sympy.Poly(self.polynomial, t)  # 根据多项式构造多项式对象
        self.poly_coefficient = polynomial.coeffs()  # 获取多项式的系数
        self.coefficient_order = polynomial.monoms()  # 多项式系数对应的最高阶次

    def cal_interp_x0(self, x0):
        '''
        计算所给定的插值点的插值，即插值
        @param x0: 所求插值的x坐标值
        @return:
        '''
        self.y0 = interp_utils.cal_interp_x0(self.polynomial, x0)
        return self.y0

    def plt_interpolation(self, x0=None, y0=None):
        '''
        可视化插值图像和所求的插值点
        @return:
        '''
        params = (self.polynomial, self.x, self.y, 'Lagrange', x0, y0)
        interp_utils.plt_interpolation(params)
