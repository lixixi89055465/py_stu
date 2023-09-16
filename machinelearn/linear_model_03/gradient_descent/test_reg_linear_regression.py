# -*- coding: utf-8 -*-
# @Time    : 2023/9/15 下午11:34
# @Author  : nanji
# @Site    : 
# @File    : test_reg_linear_regression.py
# @Software: PyCharm 
# @Comment :

import numpy as np
import matplotlib.pyplot as plt
from machinelearn.model_evaluation_selection_02.Polynomial_feature import PolynomialFeatureData
from machinelearn.linear_model_03.gradient_descent.regularization_linear_regression import \
    RegularizationLinearRegression


def objective_fun(x):
    '''
    目标函数
    :param x:
    :return:
    '''
    return 0.5 * x ** 2 + x + 2


np.random.seed(42)
n = 200  # 采样数据的样本量
raw_x = np.sort(6 * np.random.rand(n, 1) - 3, axis=0)  # [-3,3]区间，排序，二维数组n*1
raw_y = objective_fun(raw_x) + np.random.randn(n, 1)  # 二维数组

feature_obj = PolynomialFeatureData(raw_x, degree=13, with_bias=False)
X_train = feature_obj.fit_transform()  # 特征数据的构造

x_test_raw = np.linspace(-3, 3, 150)  # 测试数据
feature_obj = PolynomialFeatureData(x_test_raw, degree=13, with_bias=False)
X_test = feature_obj.fit_transform()
y_test = objective_fun(x_test_raw)

reg_ratio = [0.1, 0.5, 1, 2, 3, 5]  # 正则化系数
alpha, batch_size, max_epochs = 0.1, 10, 300
plt.figure(figsize=(15, 8))

for i, ratio in enumerate(reg_ratio):
    plt.subplot(231 + i)
    # 不采用正则化
    reg_lr = RegularizationLinearRegression(solver='form', alpha=alpha, batch_size=batch_size, max_epoch=max_epochs)
    reg_lr.fit(X_train, raw_y)
    print('NoReg, ratio = %.2f', reg_lr.get_params())
    print('=' * 100)

    y_test_pred = reg_lr.predict(X_test)  # 测试样本预测
    mse, r2, _ = reg_lr.cal_mse_r2(y_test, y_test_pred)
    plt.scatter(raw_x, raw_y, s=15, c='k')
    plt.plot(x_test_raw, y_test, lw=1.5, label='Objective Fucntion ')
    plt.plot(x_test_raw, y_test_pred, lw=1.5, label='NoReg MSE = %.5f R2= %.5f' % (mse, r2))

    # 岭回归
    ridge_lr = RegularizationLinearRegression(solver='form', alpha=alpha, batch_size=batch_size, max_epoch=max_epochs,
                                              l2_ratio=ratio)
    ridge_lr.fit(X_train, raw_y)
    print('L2Reg, ratio = %.2f' % ratio, ridge_lr.get_params())
    print('=' * 100)
    y_test_pred = ridge_lr.predict(X_test)  # 测试样本预测
    mse, r2, _ = ridge_lr.cal_mse_r2(y_test, y_test_pred)
    # plt.scatter(raw_x, raw_y, s=15, c='k')
    plt.plot(x_test_raw, y_test_pred, lw=1.5, label='L2 MSE = %.5f  R2= %.5f ' % (mse, r2))

    # LASSO回归
    lasso_lr = RegularizationLinearRegression(solver='grad', alpha=alpha, batch_size=batch_size, max_epoch=max_epochs,
                                              l1_ratio=ratio)
    lasso_lr.fit(X_train, raw_y)
    print('L1Reg, ratio = %.2f' % ratio, lasso_lr.get_params())
    print('=' * 100)
    y_test_pred = lasso_lr.predict(X_test)  # 测试样本预测
    mse, r2, _ = lasso_lr.cal_mse_r2(y_test, y_test_pred)
    # plt.scatter(raw_x, raw_y, s=15, c='k')
    plt.plot(x_test_raw, y_test_pred, lw=1.5, label='L1 MSE = %.5f  R2= %.5f ' % (mse, r2))

    # 弹性网络回归
    en_lr = RegularizationLinearRegression(solver='form', alpha=alpha, batch_size=batch_size, max_epoch=max_epochs,
                                              l2_ratio=ratio,en_rou=0.5)
    en_lr.fit(X_train, raw_y)
    print('EnReg, ratio = %.2f' % ratio, en_lr.get_params())
    print('=' * 100)
    y_test_pred = en_lr.predict(X_test)  # 测试样本预测
    mse, r2, _ = en_lr.cal_mse_r2(y_test, y_test_pred)
    # plt.scatter(raw_x, raw_y, s=15, c='k')
    plt.plot(x_test_raw, y_test_pred, lw=1.5, label='en  MSE = %.5f  R2= %.5f ' % (mse, r2))

    plt.xlabel("x", fontdict={'fontsize': 12})
    plt.ylabel("y", fontdict={'fontsize': 12})
    plt.legend(frameon=False)
    plt.axis([-3, 3, -11, 11])

plt.show()
