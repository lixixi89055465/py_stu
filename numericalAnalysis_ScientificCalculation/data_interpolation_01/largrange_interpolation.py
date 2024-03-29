# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/8/19 18:18
# @Author  : nanji
# @File    : largrange_interpolation.py
# @Description :  机器学习 https://space.bilibili.com/512662380
import numpy as np  # 数值运算，尤其是向量化的运算方式
import sympy  # 符号运算库
import matplotlib.pyplot as plt
from numericalAnalysis_ScientificCalculation.data_interpolation_01.utils import interp_utils


class LargrangeInterpolation:
    '''
    拉格朗日插值
    '''

    def __init__(self, x, y):
        '''
        拉格朗日必要参数的初始化，及各健壮性的检测
        @param x:
        @param y:
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

    def fit_interp(self):
        '''
        核心算法：生成拉格朗日插值多项式
        @return:
        '''
        # 数值运算，符号运算
        t = sympy.Symbol("t")  # 定义符号变量
        self.polynomial = 0.0  # 插值多项式实例化
        for i in range(self.n):
            # 针对每个数据点，构造插值基函数
            basis_fun = self.y[i]  # 插值基函数
            for j in range(i):
                basis_fun *= (t - self.x[j]) / (self.x[i] - self.x[j])
            for j in range(i + 1, self.n):
                basis_fun *= (t - self.x[j]) / (self.x[i] - self.x[j])
            self.polynomial += basis_fun  # 插值多项式累加
        # 插值多项式特征
        self.polynomial = sympy.expand(self.polynomial)
        polynomial = sympy.Poly(self.polynomial, t)  # 根据多项式构造多项式对象
        self.poly_coefficient = polynomial.coeffs()  # 获取多项式的系数
        # self.coefficient_order=polynomial.degree() # 多项式系数对相应次
        self.coefficient_order = polynomial.monoms()  # 多项式系数对应的最高阶次
        print(self.polynomial)

    def cal_interp_x0(self, x0):
        '''
        计算所给定的插值点的插值，即插值
        @param x0: 所求插值的x坐标值
        @return:
        '''
        self.y0=interp_utils.cal_interp_x0(self.polynomial,x0)
        return self.y0



    def plt_interpolation(self, x0=None, y0=None):
        '''
        可视化插值图像和所求的插值点
        @return:
        '''
        params = (self.polynomial, self.x, self.y, 'Lagrange',x0, y0)
        interp_utils.plt_interpolation(params)
