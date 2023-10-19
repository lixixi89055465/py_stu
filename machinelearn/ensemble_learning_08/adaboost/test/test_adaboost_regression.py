# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/10/19 21:20
# @Author  : nanji
# @File    : test_adaboost_regression.py
# @Description :

import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
from machinelearn.decision_tree_04.decision_tree_R import DecisionTreeRegression
from machinelearn.ensemble_learning_08.adaboost.adaboost_regression import AdaBoostRegression
from sklearn.metrics import r2_score

boston = load_boston()
X, y = boston.data, boston.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
base_ht = DecisionTreeRegression(max_bins=50, max_depth=5)
# abr = AdaBoostRegression(base_estimator=base_ht, n_estimators=3, comb_strategy='weight_mean')
abr = AdaBoostRegression(base_estimator=base_ht, n_estimators=3, comb_strategy='weight_median')
abr.fit(X_train, y_train)
y_hat= abr.predict(X_test)
print(r2_score(y_test, y_hat))