# -*- coding: utf-8 -*-
# @Time    : 2023/10/23 19:13
# @Author  : nanji
# @Site    : 
# @File    : test_bagging_c1.py
# @Software: PyCharm 
# @Comment :

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from machinelearn.decision_tree_04.decision_tree_R import DecisionTreeRegression
from machinelearn.decision_tree_04.decision_tree_C import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from machinelearn.ensemble_learning_08.gradient.bagging_c_r import BaggingClassifierRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

iris = load_iris()
X, y = iris.data, iris.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, shuffle=True, stratify=y)

base_es = DecisionTreeClassifier(max_bins=50, max_depth=10, is_feature_all_R=True)
bagrc = BaggingClassifierRegression(base_estimator=base_es, n_estimators=20, task='c',OOB=True)
bagrc.fit(X_train, y_train)
y_hat = bagrc.predict(x_test=X_test)
print(classification_report(y_test, y_hat))
