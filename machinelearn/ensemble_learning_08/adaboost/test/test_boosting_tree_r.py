# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/10/21 11:32
# @Author  : nanji
# @File    : test_boosting_tree_r.py
# @Description :https://www.bilibili.com/video/BV1Nb4y1s7nV/?p=74&spm_id_from=pageDriver&vd_source=50305204d8a1be81f31d861b12d4d5cf

import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from machinelearn.decision_tree_04.decision_tree_R import DecisionTreeRegression
from sklearn.tree import DecisionTreeRegressor
from machinelearn.ensemble_learning_08.boosting_tree.boostingtree_r import BoostTreeRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# boston = load_boston()
# X, y = boston.data, boston.target
# X = StandardScaler().fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# base_ht = DecisionTreeRegression(max_bins=50, max_depth=5)
# n_estimators = np.linspace(2, 32, 29, dtype=np.int)
# r2_scores = []
# for n in n_estimators:
#     btr = BoostTreeRegression(base_estimator=base_ht, n_estimators=n)
#     btr.fit(X_train, y_train)
#     y_hat = btr.predict(X_test)
#     r2_scores.append(r2_score(y_test, y_hat))
#     print(n, ":", r2_scores[-1])
# plt.figure(figsize=(7, 5))
# plt.plot(n_estimators, r2_scores, 'ko-', lw=1)
# plt.show()

# idx = np.argsort(y_test)  # 对真值做排序
# plt.figure(figsize=(7, 5))
# plt.plot(y_test[idx], 'k-', lw=1.5, label='Test true')
# plt.plot(y_hat[idx], 'r-', lw=1, label='Predict')
# plt.legend(frameon=False)
# plt.title('Regression Boosting Tree, R2=%.5f,MSE=%.5f' % \
#           (r2_score(y_test, y_hat), ((y_test - y_hat) ** 2).mean()))
# plt.xlabel("Test Samples Serial Number", fontdict={'fontsize': 12})
# plt.ylabel('True VS Predict', fontdict={'fontsize': 12})
# plt.grid(ls=':')
# plt.show()

X = np.linspace(1, 10, 10).reshape(-1, 1)
y = np.array([5.56, 5.70, 5.91, 6.40, 6.8, 7.05, 8.90, 8.7, 9.0, 9.05])
base_ht = DecisionTreeRegressor(max_depth=1)
r2_scores = []
for n in range(2, 8):
    btr = BoostTreeRegression(base_estimator=base_ht, n_estimators=n)
    btr.fit(X, y)
    y_hat = btr.predict(X)
    r2_scores.append(r2_score(y, y_hat))
    print(n, ':', r2_scores[-1], np.sum((y - y_hat) ** 2))
