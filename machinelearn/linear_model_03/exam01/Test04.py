# -*- coding: utf-8 -*-
# @Time    : 2023/10/26 16:19
# @Author  : nanji
# @Site    : 
# @File    : Test04.py
# @Software: PyCharm 
# @Comment :

import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from machinelearn.model_evaluation_selection_02.Polynomial_feature import PolynomialFeatureData
from machinelearn.linear_model_03.closed_form_sol.LinearRegression_CFSol import LinearRegressionClosedFormSol


def objective_func(x):
    return 0.5 * x ** 2 + x + 2


n = 300
raw_x = np.sort(6 * np.random.rand(n, 1) - 3)
raw_y = objective_func(raw_x) + 0.5 * np.random.randn(n, 1)
k_fold = KFold(n_splits=10, shuffle=True, random_state=42)
# k_fold = KFold(n_splits=10, shuffle=False, random_state=42)
degree = [1, 2, 4, 6, 8, 10]
plt.figure(figsize=(21, 10.5))
for i, d in enumerate(degree):
    # 生成特征多项式
    feta_obj = PolynomialFeatureData(raw_x, degree=d, with_bias=False)
    X_sample = feta_obj.fit_transform()
    train_mse, test_mse = [], []
    for j in range(1, 270):
        train_mse_, test_mse_ = 0., 0.
        for idx_train, idx_test in k_fold.split(raw_x, raw_y):
            X_train, y_train = X_sample[idx_train], raw_y[idx_train]
            X_test, y_test = X_sample[idx_test], raw_y[idx_test]
            lr_cfs = LinearRegressionClosedFormSol()
            theta = lr_cfs.fit(X_train[:j, :], y_train[:j])  # 拟合多项式
            y_test_pred = lr_cfs.predict(X_test)
            y_train_pred = lr_cfs.predict(X_train[:j, :])  # 训练样本预测
            train_mse_ = np.mean((y_train_pred.reshape(-1) - y_train[:j].reshape(-1)) ** 2)
            test_mse_ += np.mean((y_test_pred.reshape(-1) - y_test.reshape(-1)) ** 2)
        train_mse.append(train_mse_ / 10)
        test_mse.append(test_mse_ / 10)
    plt.subplot(321 + i)
    plt.plot(train_mse, 'k*', lw=1, label="Train")
    plt.plot(test_mse, 'r--', lw=1, label="Train")
    plt.title("Learning curve by degree%d" % (i))
    plt.legend(frameon=False)
    plt.grid(ls=":")
    plt.xlabel("Train Size ")
    plt.ylabel("MSE ")
    plt.axis([1,300,0,1])
plt.show()
