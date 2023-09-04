# -*- coding: utf-8 -*-
# @Time    : 2023/9/3 15:15
# @Author  : nanji
# @Site    : 
# @File    : aitken_stepwise_interpolation.py
# @Software: PyCharm 
# @Comment :
import numpy as np  # 数值运算，尤其是向量化的运算方式
import sympy  # 符号运算库
import matplotlib.pyplot as plt
from numericalAnalysis_ScientificCalculation.data_interpolation_01.utils import interp_utils


class AitkenStepwiseInterpolation:
    '''
    艾特肯逐步插值基本思想时K+1次插值多项式可由两个k次插值多项式得出
    不带精度要求，逐步递推到最后一个多项式,用次多项式进行数据插值
    '''

    def __init__(self, x, y):
        '''
        艾特肯逐步插值基本思想时K+1次插值多项式可由两个k次插值多项式得出
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
        self.aitken_mat = None

    def fit_interp(self):
        '''
        核心算法：生成艾特肯逐步插值多项式
        @return:
        '''
        # 数值运算，符号运算
        t = sympy.Symbol("t")  # 定义符号变量
        self.aitken_mat = sympy.zeros(self.n, self.n + 1)
        self.aitken_mat[:, 0], self.aitken_mat[:, 1] = self.x, self.y
        self.polynomial = 0.0  # 插值多项式实例化
        poly_next = [t for _ in range(self.n)]  # 用于存储下一列递归多项式
        poly_before = np.copy(self.y)  # 用于存储上一列递推多项式
        for i in range(self.n - 1):
            # 针对每一个数据点
            for j in range(i + 1, self.n):
                poly_next[j] = (poly_before[j] * (t - self.x[i]) - poly_before[i] * (t - self.x[j])) \
                               / (self.x[j] - self.x[i])
            # poly_before[i + 1:] = poly_next[i + 1:]  # 多项式的递推，下一列赋值给上一列
            poly_before = poly_next  # 多项式的递推，下一列赋值给上一列
            self.aitken_mat[i + 1:, i + 2] = poly_next[i + 1 :]
        # 插值多项式特征
        self.polynomial = poly_next[-1]
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
        self.y0 = interp_utils.cal_interp_x0(self.polynomial, x0)
        return self.y0

    def plt_interpolation(self, x0=None, y0=None):
        '''
        可视化插值图像和所求的插值点
        @return:
        '''
        params = (self.polynomial, self.x, self.y, 'Aitken StepWise', x0, y0)
        interp_utils.plt_interpolation(params)
