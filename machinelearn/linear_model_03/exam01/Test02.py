# -*- coding: utf-8 -*-
# @Time    : 2023/10/25 9:40
# @Author  : nanji
# @Site    : 
# @File    : Test02.py
# @Software: PyCharm 
# @Comment :

import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from machinelearn.decision_tree_04.decision_tree_R import DecisionTreeRegression
from machinelearn.decision_tree_04.decision_tree_C import DecisionTreeClassifier
from machinelearn.ensemble_learning_08.gradient.gradientboosting_r import GradientBoostRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from machinelearn.linear_model_03.exam01.LinearRegression_GradDesc import LinearRegression_GradDesc
import pandas as pd

# air = pd.read_csv('../../data/airquality.csv', encoding='GBK').dropna()
# print(air.columns)
# feature_names = ['aqi', 'pm2_5', 'pm10', 'so2', 'no2', 'co', 'o3']
# X = np.asarray(air.loc[:, feature_names], dtype=np.int)
# y = np.asarray(air.loc[:, "aqi"], dtype=np.int)
# # X = StandardScaler().fit_transform(X)
#
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=1, shuffle=True)
# # lggd_obj = LinearRegression_GradDesc(normalize=True, fit_intercept=True,  alpha=0.2, batch_size=200, epochs=300)
# lggd_obj = LinearRegression_GradDesc(normalize=True, fit_intercept=True,  alpha=0.08, batch_size=1, epochs=300)
# lggd_obj.fit(X_train, y_train, X_test, y_test)
# theta = lggd_obj.get_params()  # 获得原型系数
# print('线性模型拟合系数如下:')
# for i, fn in enumerate(feature_names):
#     print(fn + ":", theta[i])
#
# print('const :', theta[-1])
# # 模型预测，即对测试样本进行预测
# y_pred=lggd_obj.predict(X_test)
# lggd_obj.plt_predict(y_pred+10,y_test,is_sort=True)
# lggd_obj.plt_loss_curve(is_show=True)
# plt.show()
from sklearn.datasets import load_boston

# np.random.seed(42)
#
# hourse = load_boston()
# X, y = hourse.data, hourse.target
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=5, shuffle=True)
# lggd_obj = LinearRegression_GradDesc(normalize=True, alpha=0.3, batch_size=50, epochs=300)
#
# lggd_obj.fit(x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)
# y_pred = lggd_obj.predict(x_test=X_test)
# lggd_obj.plt_predict(y_pred, y_test, is_show=True)
# lggd_obj.plt_loss_curve(is_show=True)

np.random.seed(42)
X = np.random.rand(1000, 5)
coef = np.array([4.2, -2.5, 1.6, 5.1, -2.5])
y = coef.dot(X.T) + np.random.randn(1000, 1).reshape(-1) * 0.5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5, shuffle=True)
lggd_obj = LinearRegression_GradDesc(normalize=True, fit_intercept=True, alpha=0.5, batch_size=20, epochs=300)
lggd_obj.fit(X_train,y_train)
y_pred = lggd_obj.predict(x_test=X_test)
lggd_obj.plt_predict(y_pred, y_test)
lggd_obj.plt_loss_curve(is_show=True)
print('0'*100)
print(lggd_obj.get_params())
