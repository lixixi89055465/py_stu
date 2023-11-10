# -*- coding: utf-8 -*-
# @Time    : 2023/10/22 9:51
# @Author  : nanji
# @Site    : 
# @File    : test_gradboost_c1.py
# @Software: PyCharm 
# @Comment :
from sklearn.datasets import load_iris,load_digits,load_breast_cancer
from sklearn.model_selection import train_test_split
from machinelearn.decision_tree_04.decision_tree_R import DecisionTreeRegression
from sklearn.tree import DecisionTreeRegressor
from machinelearn.ensemble_learning_08.gradient.gradientboosting_c import Gradientboosting_c
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# iris = load_iris()

print('0'*100)
digits=load_digits()
# digits=load_breast_cancer()
# X, y = iris.data, iris.target
X, y = digits.data, digits.target
# X=PCA(n_components=10).fit_transform(X)
X=StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=y)
# base_es = DecisionTreeRegression(max_bins=50, max_depth=3)
base_es = DecisionTreeRegressor(max_depth=5)
gbc = Gradientboosting_c(base_estimator=base_es, n_estimators=20,learning_rate=0.9)
gbc.fit(X_train, y_train)
y_hat = gbc.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_hat))
