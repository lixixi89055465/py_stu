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
gas = pd.read_csv('../../data/gt_2011.csv').dropna()
feature_names = ['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP', 'CO', 'NOX']
print(gas.columns)
X = np.asarray(gas.loc[:, feature_names])
y = np.asarray(gas.loc[:, 'TEY'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, shuffle=True)
lgcfs_obj = LRCloseFormSol(normalize=True, fit_intercept=True)
lgcfs_obj.fit(X_train, y_train)
theta = lgcfs_obj.get_params()
print("线性回归模型拟合系数如下 !")
print(theta)
y_pred = lgcfs_obj.predict(X_test)
lgcfs_obj.plt_predict(y_pred, y_test, is_sort=True)
# 采用sklearn 库进行线性回归和预测
from sklearn.linear_model import LinearRegression

lr = LinearRegression().fit(X_train, y_train)

print(lr.intercept_)
print('1'*100)
print(lr.coef_)
y_test_predict = lr.predict(X_test)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_test_predict)
r2 = r2_score(y_test, y_test_predict)
print("均方误差与判定系数为:", mse, r2)

