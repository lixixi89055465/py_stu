# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/8/20 14:06
# @Author  : nanji
# @File    : test_poly_regression.py
# @Description :

import numpy as np
import matplotlib.pyplot as plt
from machinelearn.model_evaluation_selection_02.polynomial_regression_curve import PolynomialRegressionCurve
from machinelearn.model_evaluation_selection_02.Polynomial_feature import PolynomialFeatureData

objective_function = lambda x: 3 * np.exp(-1) * np.sin(x)  # 目标函数
np.random.seed(0)  # 随机种子
n = 10  # 样本量
raw_x = np.linspace(0, 6, n)
raw_y = objective_function(raw_x) + np.random.randn(n)  # 目标值+ 噪声，模拟真是采样数据
degrees = [1, 3, 5, 7, 10, 12]  # 多项式阶次
for i, degree in enumerate(degrees):
    feat_data = PolynomialFeatureData(raw_x, degree, with_bias=True)  # 根据阶次生成特征数据
    X_sample = feat_data.fit_transform()
    poly_obj = PolynomialRegressionCurve(X_sample, raw_y, fit_intercept=True)
    theta = poly_obj.fit()  # 闭式解求解最优参数
    print('degree : %d,theta is ' % degree, theta)
    x_test = np.linspace(0, 6, 150)  # 生成测试样本
    y_pred = poly_obj.predict(x_test)  # 预测
    # print(y_pred.shape)
    # 可视化 :采样散点图，真实目标函数，拟合的模型
    plt.scatter(raw_x, raw_y, edgecolors='k', s=16, label='Raw Data')  # 采样数据散点图
    plt.scatter(x_test, y_pred, edgecolors='b', s=16, label='Test Data')  # 采样数据散点图
    plt.show()
