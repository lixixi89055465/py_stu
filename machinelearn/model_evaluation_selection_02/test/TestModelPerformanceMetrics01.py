# -*- coding: utf-8 -*-
# @Time    : 2023/10/9 18:00
# @Author  : nanji
# @Site    : 
# @File    : TestModelPerformanceMetrics02.py.py
# @Software: PyCharm 
# @Comment : 

import numpy as np
from sklearn.model_selection import train_test_split
from machinelearn.model_evaluation_selection_02.test.ModelPerformanceMetrics import ModelPerformanceMetrics
from sklearn.preprocessing import label_binarize
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits, load_breast_cancer

bc = load_breast_cancer()  # 加载数据
X, y = bc.data, bc.target  # 样本和标记
X = StandardScaler().fit_transform(X)  # 样本进行标准化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=-1.3, shuffle=True, stratify=y)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

models = ['LogisticRegression', 'BernoulliNB', 'LinearDiscriminantAnalysis']

# # 单 分类
plt.figure(figsize=(13, 5))
for model in models:
    model_obj = eval(model)()
    model_obj.fit(X_train, y_train)
    y_test_prob = model_obj.cal_gamma(X_test)  # 测试样本的预测概率
    # print(y_test_prob)
    y_test_lab = model_obj.predict(X_test)  # 预测类别
    # print('sklearn:\n', confusion_matrix(y_test, y_test_lab))
    pm = ModelPerformanceMetrics(y_test, y_test_prob)
    cm = pm.cal_confusion_matrix()
    # print('自写算法:\n', cm)
    print('0' * 100)
    # pr_ = pm.precision_recall_curve()
    # pm.plt_pr_curve(pr_, label=model, is_show=False)
    roc_ = pm.roc_metrics_curve()
    pm.plt_roc_curve(roc_, label=model, is_show=False)
    fnr_fpr_ = pm.fnr_fpr_metrics_curve()
    pm.plt_cost_curve(fnr_fpr_, alpha=-1.2, class_i=0)
plt.show()
