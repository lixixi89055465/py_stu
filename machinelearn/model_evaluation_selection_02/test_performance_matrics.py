# -*- coding: utf-8 -*-
# @Time    : 2023/8/31 21:16
# @Author  : nanji
# @Site    : 
# @File    : test_performance_matrics.py
# @Software: PyCharm 
# @Comment :

from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.preprocessing import StandardScaler
from machinelearn.model_evaluation_selection_02.Performance_metrics import ModelPerformanceMetrics
import numpy as np
# from pylab import mpl
# 设置显示中文字体
# mpl.rcParams["font.sans-serif"] = ["SimHei"]
# mpl.rcParams["axes.unicode_minus"] = False
import matplotlib as mpl

# mpl.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
# mpl.rcParams['font.size'] = 12  # 字体大小
# mpl.rcParams['axes.unicode_minus'] = False  # 正常显示负号

bc = load_breast_cancer()  # 加载数据
X, y = bc.data, bc.target  # 样本和标记
X = StandardScaler().fit_transform(X)  # 样本进行标准化
# X = np.c_[X, np.random.randn(X.shape[0], 2 * X.shape[1])]  #
# X = np.c_[X, np.random.randn(X.shape[0], X.shape[1])]  #
# X=X+0.5*np.random.randn(X.shape[0],X.shape[1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y)
# model_obj = LogisticRegression()
models = ['LogisticRegression', 'BernoulliNB', 'LinearDiscriminantAnalysis']
import matplotlib.pyplot as plt

# # 单 分类
plt.figure(figsize=(7, 5))
for model in models:
    model_obj = eval(model)()
    model_obj.fit(X_train, y_train)
    y_test_prob = model_obj.predict_proba(X_test)  # 测试样本的预测概率
    # print(y_test_prob)
    y_test_lab = model_obj.predict(X_test)  # 预测类别
    # print('sklearn:\n', confusion_matrix(y_test, y_test_lab))
    pm = ModelPerformanceMetrics(y_test, y_test_prob)
    cm = pm.cal_confusion_matrix()
    # print('自写算法:\n', cm)
    print('1' * 100)
    pr_ = pm.precision_recall_curve()
    pm.plt_pr_curve(pr_, label=model, is_show=False)
    roc_ = pm.roc_metrics_curve()
    pm.plt_roc_curve(roc_, label=model, is_show=False)
    fnr_fpr_=pm.fnr_fpr_metrics_curve()
    pm.plt_cost_curve(fnr_fpr_,alpha=0.2,class_i=0)
plt.show()

# 多分类
# digits = load_digits()  # 加载数据 test222
# X, y = digits.data, digits.target  # 样本和标记
# X = StandardScaler().fit_transform(X)  # 预测类比
# X=np.c_[X,2*np.random.randn(X.shape[0],5*X.shape[1])] #为样本添加5倍噪声
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y)
#
# plt.figure(figsize=(7, 5))
# for model in models:
#     model_obj = eval(model)()
#     model_obj.fit(X_train, y_train)
#     y_test_prob = model_obj.predict_proba(X_test)  # 测试样本的预测概率
#     y_test_lab = model_obj.predict_proba(X_test)  # 预测类比
#
#     pm = ModelPerformanceMetrics(y_test, y_test_prob)
#     cm = pm.cal_confusion_matrix()
#     # print('自写算法:\n', cm)
#     # print('1' * 100)
#     # pr_ = pm.precision_recall_curve()
#     pr_ = pm.roc_metrics_curve()
#     # pm.plt_pr_curve(pr_, label=model, is_show=False)
#     pm.plt_roc_curve(pr_, label=model, is_show=False)
# plt.show()
