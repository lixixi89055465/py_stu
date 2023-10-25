# -*- coding: utf-8 -*-
# @Time    : 2023/10/25 13:13
# @Author  : nanji
# @Site    : 
# @File    : Test03.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import matplotlib.pyplot as plt
from machinelearn.model_evaluation_selection_02.Polynomial_feature \
    import PolynomialFeatureData
from machinelearn.linear_model_03.closed_form_sol.LinearRegression_CFSol \
    import LinearRegressionClosedFormSol


def objective_func(x):
    return 0.5 * x ** 2 + x + 2


np.random.seed(42)

n = 30

raw_x = np.sort(np.random.rand(n, 1) * 6 - 3, axis=0)
raw_y = objective_func(raw_x) + np.random.rand(n, 1) * .5
degree = [1, 2, 5, 10, 15, 20]
plt.figure(figsize=(15, 8))
for i, d in enumerate(degree):
    feature_obj = PolynomialFeatureData(raw_x, d, with_bias=False)
    X_sample = feature_obj.fit_transform()
    lr_cfg = LinearRegressionClosedFormSol()
    lr_cfg.fit(X_sample, raw_y)
    y_train_pred = lr_cfg.predict(X_sample)
    theta = lr_cfg.get_params()
    print('degree : %d ,theta is ' % d, theta[0].reshape(-1), theta[1])
    # 测试样本采用
    x_test_raw = np.linspace(-3, 3, 150)
    y_test = objective_func(x_test_raw)
    feature_obj = PolynomialFeatureData(x_test_raw, d, with_bias=False)
    x_test = feature_obj.fit_transform()
    y_test_pred = lr_cfg.predict(x_test)
    # 可视化多项式拟合曲线
    plt.subplot(231 + i)
    plt.scatter(raw_x, raw_y, edgecolors='k', s=15, label='Raw Data')
    plt.plot(x_test_raw, y_test, 'k-', lw=1, label="Objective Fun")
    plt.plot(x_test_raw, y_test_pred, 'k-', lw=1, label="Polynomial Fun")
    plt.legend(frameon=False)
    plt.grid(ls=":")
    plt.xlabel("$X$", fontdict={"fontsize": 12})
    plt.ylabel("$Y$", fontdict={"fontsize": 12})
    test_ess = (y_test_pred.reshape(-1) - y_test) ** 2
    score_mse, score_std = np.mean(test_ess), np.std(test_ess)
    train_ess = (y_train_pred - raw_y) ** 2  # 训练误差平方和
    train_mse, train_std = np.mean(train_ess), np.std(train_ess)
    plt.title("Degree {} Tes MSE = {:.2e} (+/- {:.2e})\n Train MSE={:.2e}(+/-{:.2e}". \
              format(d, score_mse, score_std, train_mse, train_std))
    plt.axis([-3, 3, 0, 9])
plt.show()
