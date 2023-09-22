# -*- coding: utf-8 -*-
# @Time    : 2023/9/15 16:55
# @Author  : nanji
# @Site    : 
# @File    : tsetP-R-curve.py
# @Software: PyCharm 
# @Comment : 3. 性能度量——P-R曲线代码示例，sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


def data_preproce():
    # 加载数据，数据预处理
    digits = datasets.load_digits()
    X, y = digits.data, digits.target
    random_state = np.random.RandomState()
    n_samples, n_class = X.shape
    # X = np.c_[X, random_state.randn(n_samples, 10 * n_class)]  # 添加噪声特征
    X = StandardScaler().fit_transform(X)
    y = label_binarize(y, classes=np.unique(y))  # one-hot
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True, stratify=y)
    return X_train, X_test, y_train, y_test


def model_train(model):
    # 模型训练
    classifier = OneVsRestClassifier(model)
    classifier.fit(X_train, y_train)
    y_score = classifier.decision_function(X_test)
    return y_score


def micro_PR(y_test, y_score):
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes = y_score.shape[1]  # 类别数
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
    precision['micro'], recall['micro'], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
    average_precision['micro'] = average_precision_score(y_test, y_score, average='micro')
    return precision, recall, average_precision


def plt_PR_curve(precision, recall, average_precision, label):
    # 绘制P-R 曲线
    label = label + ": AP={0:0.2f}".format(average_precision['micro'])
    plt.step(recall['micro'], precision['micro'], where='post', lw=2, label=label)


X_train, X_test, y_train, y_test = data_preproce()
# y_score = model_train(LogisticRegression())
# precision, recall, average_precision = micro_PR(y_test, y_score)
# plt.figure(figsize=(12, 8))
# plt_PR_curve(precision, recall, average_precision, "LogisiticREgression")
#
# from sklearn.svm import SVC

# y_score = model_train(SVC())
# precision, recall, average_precision = micro_PR(y_test, y_score)
# plt_PR_curve(precision, recall, average_precision, "svm.SVC")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# y_score = model_train(LinearDiscriminantAnalysis())
# precision, recall, average_precision = micro_PR(y_test, y_score)
# plt_PR_curve(precision, recall, average_precision, " LinearDiscriminantAnalysis")

# plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
# plt.xlabel('Recall', fontsize=12)
# plt.ylabel('Precision', fontsize=12)
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.grid()
# plt.title('Average precision score,micro-averaged over all classes', fontsize=14)
# plt.legend(fontsize=12)
# plt.show()

from itertools import cycle

y_score = model_train(LogisticRegression())
precision, recall, average_precision = micro_PR(y_test, y_score)
plt.figure(figsize=(9, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines, labels = [], []  #
# F-score 等高线绘制
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    line, = plt.plot(x[y >= 0], y[y >= 0], color='gray', ls='--', alpha=0.5)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(line)
labels.append('micros-average(area={0:0.2f}'.format(average_precision['micro']))

for i in range(y_score.shape[1]):
    line, = plt.plot(recall[i], precision[i], lw=1.5)
    lines.append(line)
    labels.append('class {0} ( area = {1:0.2f})'.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Extension of precision-Recall curve to multi-class', fontsize=14)
plt.legend(lines, labels, loc=(1.02, 0), prop=dict(size=12))

plt.show()
