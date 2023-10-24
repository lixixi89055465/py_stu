# -*- coding: utf-8 -*-
# @Time    : 2023/10/24 9:24
# @Author  : nanji
# @Site    : 
# @File    : TestLRCloseFormSol02.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot  as plt
from machinelearn.linear_model_03.exam01.LRCloseFormSol import LRCloseFormSol
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from machinelearn.ensemble_learning_08.gradient.gradientboosting_r import GradientBoostRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.random.rand(1000, 5)

coef = np.array([4.2, -2.5, 1.6, 5.1, -2.5])
y = coef.dot(X.T)+0.5*np.random.randn(1000)
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5, shuffle=True)

lgcfs_obj = LRCloseFormSol(normalize=True, fit_intercept=True)
lgcfs_obj.fit(X_train, y_train)

theta = lgcfs_obj.get_params()
print("线性回归模型拟合系数与截距 ")
print(theta)
y_pred = lgcfs_obj.predict(X_test)
# 模型预测，即对测试样本进行预测
lgcfs_obj.plt_predict(y_pred, y_test, is_sort=True)
mse, r2, r2_adj = lgcfs_obj.cal_mse_r2(y_pred, y_test)
from sklearn.linear_model import LinearRegression

lr = LinearRegression().fit(X_train, y_train)
print('截距:', lr.intercept_)
print('系数:', lr.coef_)
y_test_predict = lr.predict(X_test)
from sklearn.metrics import mean_squared_error

print('0' * 100)
mse = mean_squared_error(y_test, y_test_predict)
r2 = r2_score(y_test, y_test_predict)
print("均方误差与判定系数为：", mse, r2)

