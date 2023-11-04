# -*- coding: utf-8 -*-
# @Time    : 2023/10/13 9:48
# @Author  : nanji
# @Site    : 
# @File    : TestP68_1.py
# @Software: PyCharm 
# @Comment : 

import pandas as pd
import sklearn.datasets as sds
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB
from machinelearn.model_evaluation_selection_02.test.ModelPerformanceMetrics import ModelPerformanceMetrics
import matplotlib.pyplot as plt

digits = sds.load_digits()  # 多分类
X, y = digits.data, digits.target
print("手写数字原本样本量和特征数 ", X.shape)
np.random.seed(0)
X = StandardScaler().fit_transform(X)
X = np.c_[X, np.random.randn(X.shape[0], 5 * X.shape[1])]  # 添加噪声
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True, stratify=y)
models = ['LogisticRegression', 'BernoulliNB', 'LinearDiscriminantAnalysis']
plt.figure(figsize=(14, 5))
for model in models:
    m_obj = eval(model)()
    m_obj.fit(X_train, y_train)
    y_test_prob = m_obj.cal_gamma(X_test)
    pm = ModelPerformanceMetrics(y_test, y_test_prob)
    plt.subplot(121)
    pr = pm.precision_recall_curve()
    pm.plt_pr_value(pr, label=model, is_show=False)
    plt.subplot(122)
    roc = pm.roc_metrics_curve()
    pm.plt_roc_curve(roc, label=model, is_show=False)
plt.show()

from sklearn.metrics import confusion_matrix, classification_report

lg_obj = LogisticRegression()
lg_obj.fit(X_train, y_train)
y_test_pred = lg_obj.predict(X_test)
y_test_prob = lg_obj.predict_proba(X_test)
print(confusion_matrix(y_test, y_test_pred))
print('=' * 100)
pm = ModelPerformanceMetrics(y_test, y_test_prob)
cm = pm.cal_confusion_matrix()
print(cm)
print('=' * 100)
print(classification_report(y_test, y_test_pred))
print('=' * 100)
print(pm.cal_classification_report())
