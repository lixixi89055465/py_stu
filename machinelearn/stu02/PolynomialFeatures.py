# -*- coding: utf-8 -*-
# @Time    : 2023/8/31 9:56
# @Author  : nanji
# @Site    : 
# @File    : PolynomialFeatures.py
# @Software: PyCharm 
# @Comment :
import numpy as np
class PolynomialFeatures:
    '''
    生产特征多项式数据
    '''
    def __init__(self, x, degree, with_bias=False):
        self.x = x
        self.degree = degree
        self.with_bias = with_bias
        if with_bias:
            self.data = np.zeros((len(x), degree + 1))
        else:
            self.data = np.zeros((len(x), degree))

    def fit_transform(self):
        '''
        生产特征多项式数据
        :return:
        '''
        if self.with_bias:
            for i in range(self.degree + 1):
                self.data[:, i] = (self.x ** i).reshape(-1)
        else:
            for i in range(self.degree):
                self.data[:, i] = (self.x ** (i + 1)).reshape(-1)
        return self.data

