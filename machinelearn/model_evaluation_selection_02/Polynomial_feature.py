# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/8/19 12:24
# @Author  : nanji
# @File    : Polynomial_feature.py
# @Description :  机器学习 https://space.bilibili.com/512662380
import numpy as np


class PolynomialFeatureData:
    def __init__(self, x, degree, with_bias=False):
        '''
        参数初始化
        @param x: 采样数据,向量形式
        @param degree: 多项式最高阶次
        @param with_bias: 是否需要偏置项
        '''
        self.x = np.asarray(x)
        self.degree = degree
        self.with_bias = with_bias
        if with_bias:
            self.data = np.zeros((len(x), degree + 1))
        else:
            self.data = np.zeros((len(x), degree))

    def fit_transform(self):
        '''
        构造多项式特征数据
        @return:
        '''
        if self.with_bias:
            self.data[:,0] = np.ones(len(self.x)).reshape(-1)
            self.data[:, 1] = self.x.reshape(-1)
            for i in range(2, self.degree + 1):
                self.data[:, i] = (self.x ** i).reshape(-1)
        else:
            for i in range(self.degree):
                self.data[:, i] = (self.x ** (i + 1))
        return self.data



# if __name__ == '__main__':
#     x = np.random.randn(5)
#     # feature = PolynomialFeatureData(x, 3, True)
#     feature = PolynomialFeatureData(x, 3, False)
#     data=feature.fit_transform()
#     print(data)