# -*- coding: utf-8 -*-
# @Time    : 2023/10/28 13:16
# @Author  : nanji
# @Site    : 
# @File    : test_rf_c1.py
# @Software: PyCharm 
# @Comment :https://www.bilibili.com/video/BV1Nb4y1s7nV/?p=82&spm_id_from=444.41.top_right_bar_window_history.content.click&vd_source=50305204d8a1be81f31d861b12d4d5cf

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from machinelearn.decision_tree_04.decision_tree_R import DecisionTreeRegression
# from machinelearn.decision_tree_04.decision_tree_C import DecisionTreeClassifier
from machinelearn.ensemble_learning_08.gradient.bagging_c_r import BaggingClassifierRegression
from machinelearn.ensemble_learning_08.randomforest.rf_classifier_regressor import RandomForestClassifierRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns
import pandas as pd

# iris = load_iris()
# X, y = iris.data, iris.target
# wine= load_wine()
# X, y = wine.data, wine.target
digit = load_digits()
X, y = digit.data, digit.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split( \
    X, y, test_size=0.25, random_state=42, shuffle=True, stratify=y)

# base_es = DecisionTreeClassifier(max_bins=50, max_depth=10, is_feature_all_R=True)
base_es = DecisionTreeClassifier(max_depth=10)
rf_model = RandomForestClassifierRegressor(base_estimator=base_es, \
                                           n_estimators=30, task='c', OOB=True, feature_importance=True)
rf_model.fit(X_train, y_train)
y_hat = rf_model.predict(x_test=X_test)
print(classification_report(y_test, y_hat))
print('包外估计的精度：', rf_model.oob_score)
print('特征重要性评分：', rf_model.feature_importance_scores)

plt.figure(figsize=(8, 5))
data_pd = pd.DataFrame([digit.feature_names, rf_model.feature_importance_scores]).T
data_pd.columns = ['Feature Names', 'Importance']
sns.barplot(x='Importance', y='Feature Names', data=data_pd)
plt.title("wine DataSet Feature Importance Scores", fontdict={'fontsize': 14})
plt.grid(ls=":")
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split( \
    X, y, test_size=0.25, random_state=42, shuffle=True, stratify=y)

# base_es = DecisionTreeClassifier(max_bins=50, max_depth=10, is_feature_all_R=True)
base_es = DecisionTreeClassifier(max_depth=10)
rf_model = RandomForestClassifierRegressor( \
    base_estimator=base_es, n_estimators=30, task='c', OOB=True, feature_importance=True, \
    feature_sampling_rate=0.2)
rf_model.fit(X_train, y_train)
y_hat = rf_model.predict(x_test=X_test)
print(classification_report(y_test, y_hat))
print('包外估计的精度：', rf_model.oob_score)
print('特征重要性评分：', rf_model.feature_importance_scores)

plt.figure(figsize=(8, 5))
data_pd = pd.DataFrame([digit.feature_names, rf_model.feature_importance_scores]).T
data_pd.columns = ['Feature Names', 'Importance']
sns.barplot(x='Importance', y='Feature Names', data=data_pd)
plt.title("wine DataSet Feature Importance Scores", fontdict={'fontsize': 14})
plt.grid(ls=":")
plt.show()
