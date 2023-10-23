# -*- coding: utf-8 -*-
# @Time    : 2023/10/23 17:51
# @Author  : nanji
# @Site    : 
# @File    : TestLRCloseFormSol.py.py
# @Software: PyCharm 
# @Comment : 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from machinelearn.linear_model_03.exam01.LRCloseFormSol import LRCloseFormSol
import matplotlib.pyplot as plt

boston = load_boston()  # 加载波士顿房价数据集
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, shuffle=True)
lgcfs_obj = LRCloseFormSol(normalize=True, fit_intercept=True)
lgcfs_obj.fit(X_train, y_train)
theta = lgcfs_obj.get_params()
print('0' * 100)
for i, fn in enumerate(boston.feature_names):
    print(fn + ":", theta[i])
print('Const :', theta[-1])

# 模型预测，即对测试样本惊醒预测
y_pred = lgcfs_obj.predict(x_test=X_test)
lgcfs_obj.plt_predict(y_pred, y_test, is_sort=True)
plt.show()
