# -*- coding: utf-8 -*-
# @Time    : 2023/8/31 10:22
# @Author  : nanji
# @Site    : 
# @File    : PolynomialCurveFit.py
# @Software: PyCharm 
# @Comment :
import numpy as np
from machinelearn.stu02.PolynomialFeatures import PolynomialFeatures


class PolynomialCurveFit:
    def __init__(self, X, y, fit_intercept=False):
        self.X, self.y = X, y.reshape(-1, 1)  # 样本数据和目标数据
        self.fit_intercept = fit_intercept  # 截距项
        self.theta = None  # 回归系数

    def fit(self):
        xtx = np.dot(self.X.T, self.X) + 0.01 * np.eye(self.X.shape[-1])  # 添加到正则项，避免不可逆
        self.theta = np.dot(np.dot(np.linalg.inv(xtx), self.X.T), self.y)  # 回归公式
        return self.theta

    def predict(self, x_pre):
        '''
        预测，x_pre 为向量数据
        :param X_pre:
        :return:
        '''
        x_pre = x_pre[:, np.newaxis]
        if x_pre.shape[1] != self.X.shape[1]:
            if self.fit_intercept:
                poly_feat = PolynomialFeatures(x_pre, self.X.shape[1] - 1, with_bias=True)
                x_pre = poly_feat.fit_transform()
            else:
                poly_feat = PolynomialFeatures(x_pre, self.X.shape[1], with_bias=False)
                x_pre = poly_feat.fit_transform()
        if self.theta is None:
            self.fit()
        y_pred = np.dot(self.theta.T, x_pre.T)
        return y_pred.reshape(-1)


import matplotlib.pyplot as plt

object_fun = lambda x: 3 * np.exp(-x) * np.sin(x)
np.random.seed(0)
n = 10  #
raw_x = np.linspace(0, 6, n)
raw_y = object_fun(raw_x) + np.random.randn(n) * 0.1  #
degree = [1, 3, 5, 7, 10, 12]
for i, d in enumerate(degree):
    feat_obj=PolynomialFeatures(raw_x,d,with_bias=True)
    X_sample=feat_obj.fit_transform()
    poly_curve=PolynomialCurveFit(X_sample,raw_y,fit_intercept=True)
    theta=poly_curve.fit()
    print('degree: %d, theta is ' % d, theta.reshape(-1))
    x_test = np.linspace(0, 6, 150)
    y_pred = poly_curve.predict(x_test)

    # 可视化多项式组合曲线
    plt.subplot(231 + i)
    plt.scatter(raw_x, raw_y, edgecolors='k',label='Raw data')  # 添加噪声的采样点
    plt.plot(x_test, object_fun(x_test), 'k-', lw=1,label='object fun')
    plt.plot(x_test, y_pred, 'r--', lw=1.5, label='Polynomial Fit')
    plt.legend(frameon=False)
    plt.grid(ls=':')
    plt.xlabel('$X$', fontdict={'fontsize': 12})
    plt.ylabel('$Y$', fontdict={'fontsize': 12})
    test_ess = (y_pred - object_fun(x_test)) ** 2  # 误差平方和
    score_mse, score_std = np.mean(test_ess), np.std(test_ess)
    train_ess = np.mean((poly_curve.predict(raw_x) - raw_y) ** 2)
    plt.title('Degree {} Test_MSE ={:.2e}(+/-) {:.2e})\n Train_MSE ={:.2e}'.
              format(d, score_mse, score_std, train_ess), fontdict={'fontsize': 12})
plt.show()
