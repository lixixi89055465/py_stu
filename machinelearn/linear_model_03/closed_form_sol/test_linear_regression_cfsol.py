# -*- coding: utf-8 -*-
# @Time    : 2023/9/9 18:11
# @Author  : nanji
# @Site    : 
# @File    : test_linear_regression_cfsol.py
# @Software: PyCharm 
# @Comment :
import numpy as np
from sklearn.datasets import load_boston
from machinelearn.linear_model_03.closed_form_sol.LinearRegression_CFSol import LinearRegressionClosedFormSol
from sklearn.model_selection import train_test_split
import pandas as pd

# boston = load_boston()  # 加载数据
# X, y = boston.data, boston.target  # 样本数据和目标值
# np.random.seed(42)
# X=np.random.rand(1000,6)#随机样本值
# coeff=np.array([4.2,-2.5,7.8,3.7,-2.9,1.87])#模型系数
# y=coeff.dot(X.T)+0.5*np.random.randn(1000)#目标函数值+ 噪声
# print(coeff)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, shuffle=True)

mpg=pd.read_csv('../../data/mpg.csv').dropna()
X,y=np.asarray(mpg.loc[:,'horsepower']),np.asarray(mpg.loc[:,'displacement'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, shuffle=True)

lr_cfs = LinearRegressionClosedFormSol()  # 使用默认设置偏置项和标准化
lr_cfs.fit(X_train, y_train)
theta = lr_cfs.get_params()

# feature_names = boston.feature_names
# for i, fn in enumerate(feature_names):
#     print(fn, ':', theta[i])
#
# print('Const:', theta[-1])
print('3'*100)
print(theta)

print('0' * 100)
print(theta)
y_test_pred = lr_cfs.predict(x_test=X_test)
print('1' * 100)
# print(y_test_pred)
print('2' * 100)
mse, r2, r2_adj = lr_cfs.cal_mse_r2(y_test_pred, y_test)
print('均方误差：%.5f,判决系数:%.5f,修正判决系数:%.5f' % (mse, r2, r2_adj))
# lr_cfs.plt_predict(y_test,y_test_pred)
lr_cfs.plt_predict(y_test, y_test_pred, is_sort=True)

import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))
plt.plot(X_test,y_test,'ro',label='Test Samples')
plt.plot(X_test,y_test_pred,'k-',lw=1.5,label='Fit test sample' )
plt.legend(frameon=False)
plt.show()
