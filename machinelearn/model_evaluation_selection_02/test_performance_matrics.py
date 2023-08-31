# -*- coding: utf-8 -*-
# @Time    : 2023/8/31 21:16
# @Author  : nanji
# @Site    : 
# @File    : test_performance_matrics.py
# @Software: PyCharm 
# @Comment :

from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.preprocessing import StandardScaler
from machinelearn.model_evaluation_selection_02.ModelPerformanceMetrics import ModelPerformanceMetrics

# digits = load_breast_cancer()  # 加载数据
# X, y = digits.data, digits.target  # 样本和标记
# X = StandardScaler().fit_transform(X)  # 样本进行标准化
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y)
# lg_obj = LogisticRegression()
# lg_obj.fit(X_train, y_train)
# y_test_prob = lg_obj.predict_proba(X_test)  # 测试样本的预测概率
# print(y_test_prob)
# y_test_lab = lg_obj.predict(X_test)  # 预测类别
# print('sklearn:\n', confusion_matrix(y_test, y_test_lab))
# pm = ModelPerformanceMetrics(y_test, y_test_prob)
# cm = pm.cal_confusion_matrix()
# print('自写算法:\n', cm)
print('1' * 100)

# 多分类
digits = load_digits()  # 加载数据 test222
X, y = digits.data, digits.target  #  样本和标记
X = StandardScaler().fit_transform(X)  # 样本进行标准化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y)

lg_obj = LogisticRegression()
lg_obj.fit(X_train, y_train)
y_test_prob = lg_obj.predict_proba(X_test)  # 测试样本的预测概率
y_test_lab = lg_obj.predict(X_test)  # 预测类别
print('sklearn: \n', confusion_matrix(y_test, y_test_lab))
pm = ModelPerformanceMetrics(y_test, y_test_prob )
cm = pm.cal_confusion_matrix()
print('自写算法:  \n', cm)
