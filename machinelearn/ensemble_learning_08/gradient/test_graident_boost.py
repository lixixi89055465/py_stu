# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/10/21 16:44
# @Author  : nanji
# @File    : test_graident_boost.py
# @Description : https://www.bilibili.com/video/BV1Nb4y1s7nV/?p=76&spm_id_from=pageDriver&vd_source=50305204d8a1be81f31d861b12d4d5cf


import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
# from machinelearn.decision_tree_04.decision_tree_R import DecisionTreeRegression
from sklearn.tree import DecisionTreeRegressor
from machinelearn.ensemble_learning_08.gradient.gradientboosting_r import GradientBoostRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

boston = load_boston()
X, y = boston.data, boston.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
loss_funs = ['lae', 'huber', 'quantile', 'logcosh']
# base_es = DecisionTreeRegression(max_bins=50, max_depth=5)
base_es = DecisionTreeRegressor(max_depth=5)
for i, loss in enumerate(loss_funs):
    gbr = GradientBoostRegression(base_estimator=base_es, n_estimators=20, loss=loss)
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)
    idx = np.argsort(y_test)
    plt.subplot(221 + i)
    plt.plot(y_test[idx], 'k-', lw=1.5, label='Test True Values')
    plt.plot(y_pred[idx], 'r-', lw=1, label='Predictive Values')
    plt.legend(frameon=False)
    plt.xlabel('Observation serial number ', fontdict={'fontsize': 12})
    plt.ylabel('Test True VS Predictive Values ', fontdict={'fontsize': 12})
    plt.title('Boston Hourse Price (R2=%.5f,MSE = %.5f, loss=%s) ' % \
              (r2_score(y_test, y_pred), ((y_test - y_pred) ** 2).mean(), loss))
    plt.grid(ls=':')
plt.show()
