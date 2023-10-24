# -*- coding: utf-8 -*-
# @Time    : 2023/10/23 19:13
# @Author  : nanji
# @Site    : 
# @File    : test_bagging_c1.py
# @Software: PyCharm 
# @Comment :

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from machinelearn.decision_tree_04.decision_tree_R import DecisionTreeRegression
from machinelearn.decision_tree_04.decision_tree_C import DecisionTreeClassifier
from machinelearn.ensemble_learning_08.gradient.bagging_c_r import BaggingClassifierRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd

nursery = pd.read_csv("../../data/nursery.csv").dropna()
X, y = np.asarray(nursery.iloc[:, :-1]), np.asarray(nursery.iloc[:, -1])
y = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

base_es = DecisionTreeClassifier(max_depth=10)
bagrc = BaggingClassifierRegression(base_estimator=base_es, n_estimators=30, task='c')
bagrc.fit(X_train, y_train)
y_hat = bagrc.predict(x_test=X_test)
print(classification_report(y_test, y_hat))
