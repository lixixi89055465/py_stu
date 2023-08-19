# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/8/19 16:42
# @Author  : nanji
# @File    : polynomial_regression_curve.py
# @Description :  机器学习 https://space.bilibili.com/512662380

import numpy as np


class PolynomialRegressionCurve:
    '''
    多项式曲线拟合，采用线性回归的方法,且是闭式解
    '''

    def __init__(self, X, y, fit_intercept=False):
        '''
        参数的初始化
        @param X: 样本数据，矩阵形式的
        @param y:  目标值，向量
        @param fit_intercept:  是否你和截距 ，偏置项
        '''
        self.X, self, y = np.asarray(X), np.asarray(y)
        self.fit_intercept = fit_intercept
        self.theta = None  # 模型拟合的最优参数

    def fit(self):
        '''
        采用线性回归
        @return:
        '''
        # pinv()伪逆
        xtx = np.dot(self.X.T, self.X) + 0.01 * np.eye(self.X.shape[1])  # 添加正则项，保证矩阵是正定可逆的
        self.theta = np.linalg.inv(xtx).dot(self.X.T).dot(self.y)

    def predict(self, x_test):
        '''
        模型预测
        @param x_test:
        @return:
        '''
        from model_evaluation_selection_02.PolynomialFeatureData import PolynomialFeatureData
        x_test = x_test[:, np.newaxis]
        if x_test.shape[1] != self.X.shape[1]:
            if self.fit_intercept:
                feat_obj = PolynomialFeatureData(x_test, self.X.shape[1] - 1, with_bias=True)
                x_test=feat_obj.fit_transform()
            else:
                feat_obj=PolynomialFeatureData(x_test,self.X.shape[1],with_bias=False)
                x_test=feat_obj.fit_transform()
        if self.theta is None:
            self.fit()
        y_pred=feat_obj.fit_transform()

