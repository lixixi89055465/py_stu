# -*- coding: utf-8 -*-
# @Time    : 2023/10/27 16:51
# @Author  : nanji
# @Site    : 
# @File    : Test07.py
# @Software: PyCharm 
# @Comment :

import numpy as np
import matplotlib.pyplot as plt

from machinelearn.linear_model_03.gradient_descent.regularization_linear_regression \
    import RegularizationLinearRegression
from machinelearn.model_evaluation_selection_02.polynomial_regression_curve \
    import PolynomialFeatureData  # 构造特征数据

objective_fun = lambda x: 0.5 * x ** 2 + x + 2  # 目标函数
np.random.seed(42)
n = 30  # 样本量
# n = 200  # 样本量
raw_x = np.sort(np.random.rand(n, 1) * 6 - 3, axis=0)
raw_y = objective_fun(raw_x) + np.random.randn(n, 1)  #

enhance_dim=23
feature_obj = PolynomialFeatureData(raw_x, enhance_dim, with_bias=False)
X_samples = feature_obj.fit_transform()
reg_ratio = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]  # 正则化系数
alpha, batch_size, epochs = 0.1, 10, 300
X_test_raw = np.linspace(-3, 3, 150)  #
feature_obj = PolynomialFeatureData(X_test_raw, degree=enhance_dim, with_bias=False)
X_test = feature_obj.fit_transform()
plt.figure(figsize=(16, 10))
for i, ratio in enumerate(reg_ratio):
    plt.subplot(231 + i)
    reg_lr = RegularizationLinearRegression(alpha=alpha, batch_size=batch_size)
    reg_lr.fit(X_samples, raw_y)
    print('GD,ratio=0.00', reg_lr.get_params())
    y_test_pred = reg_lr.predict(X_test)  # 模型预测值
    plt.plot(X_test_raw, objective_fun(X_test_raw), 'k-', lw=1.5, label='Objective fun')
    mse, r2, _ = reg_lr.cal_mse_r2(objective_fun(X_test_raw), y_test_pred)
    plt.plot(X_test_raw, y_test_pred, lw=1.5, \
             label='NoReg MSE =%.5f,R2=%.5f' % (mse, r2)
             )
    lasso = RegularizationLinearRegression(l1_ratio=ratio, alpha=alpha, \
                                           batch_size=batch_size)
    lasso.fit(X_samples, raw_y)
    print('L1,ratio = %.2f:' % ratio, lasso.get_params())
    y_test_pred = lasso.predict(X_test)
    mse, r2, _ = reg_lr.cal_mse_r2(objective_fun(X_test_raw), y_test_pred)
    plt.plot(X_test_raw, y_test_pred, lw=1.5, \
             label='L1 MSE = %.5f,R2=%.5f' % (mse, r2),
             )
    # ridge = RegularizationLinearRegression(l2_ratio=ratio, alpha=alpha, \
    #                                        batch_size=batch_size)  # ridge正则化
    # ridge.fit(X_samples, raw_y)
    # print('L2,ratio =%.2f:' % ratio, ridge.get_params())
    # y_test_pred = ridge.predict(X_test)
    # mse, r2, _ = reg_lr.cal_mse_r2(objective_fun(X_test_raw), y_test_pred)
    # plt.plot(X_test_raw, y_test_pred, lw=1.5, \
    #          label='L2M3=%.5f,R2=%.5f' % (mse, r2),
    #          )
    # elastic_net = RegularizationLinearRegression(l1_ratio=ratio, l2_ratio=ratio, en_rou=0.5, \
    #                                              alpha=alpha, batch_size=batch_size)
    # elastic_net.fit(X_samples, raw_y)
    # print('EN,ratio =%.2f:' % ratio, elastic_net.get_params())
    # y_test_pred = elastic_net.predict(X_test)
    # mse, r2, _ = reg_lr.cal_mse_r2(objective_fun(X_test_raw), y_test_pred)
    # plt.plot(X_test_raw, y_test_pred, lw=1.5, \
    #          label='EN_MSE = %.5f, R2 = %.5f' % (mse, r2),
    #          )
    # plt.xlabel(r'x', fontdict={'fontsize': 12})
    # plt.ylabel(r'y', fontdict={'fontsize': 12})
    # plt.legend(frameon=False)
    # plt.grid(ls=":")
    # plt.title('closed Form_Solution with $\lambda $= %.2f' % ratio, fontdict={'fontsize': 12})
    # print('-' * 100)
plt.show()
