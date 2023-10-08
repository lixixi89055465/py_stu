# -*- coding: utf-8 -*-
# @Time    : 2023/10/7 17:23
# @Author  : nanji
# @Site    : 
# @File    : testBinaryROC.py
# @Software: PyCharm 
# @Comment :  3. 性能度量——ROC与AUC+多分类
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import random
from sklearn.multiclass import OneVsRestClassifier

import sklearn.datasets as datasets
from sklearn.preprocessing import label_binarize


def data_preprocess():
    pass


def data_preprocess():
    digits = datasets.load_digits()
    X, y = digits.data, digits.target
    rs = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, rs.randn(n_samples, 10 * n_features)]  # 添加噪声数据
    X = StandardScaler().fit_transform(X)#标准化
    y = label_binarize(y, classes=np.unique(y))  # one-hot
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0,shuffle=True, stratify=y)
    return X_train, X_test, y_train, y_test


def model_train(model):
    # 模型训练
    classifier = OneVsRestClassifier(model,n_jobs=-1)
    classifier.fit(X_train, y_train)
    y_score = classifier.predict(X_test)
    return y_score


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

X_train, X_test, y_train, y_test = data_preprocess()
y_score = model_train(LogisticRegression())  # 逻辑回归


def roc_metric(y_test, y_score):
    # 计算 orc,auc指标
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())
    auc_val = auc(fpr, tpr)
    return fpr, tpr, auc_val


import matplotlib.pyplot as plt

fpr, tpr, auc_val = roc_metric(y_test, y_score)
plt.figure(figsize=(8, 6))


def plt_ROC(fpr, tpr, auc_val, label):
    # 绘制 ROC曲线
    label = label + ': auc={0:0.2f}'.format(auc_val)
    plt.plot(fpr, tpr, label=label, lw=2)


plt_ROC(fpr, tpr, auc_val, "LogisiticRegression")

from sklearn.svm import SVC

y_score = model_train(SVC(probability=True))  # 支持向量机
fpr, tpr, auc_val = roc_metric(y_test, y_score)
plt_ROC(fpr, tpr, auc_val, 'svm.SVC')

from sklearn.tree import DecisionTreeClassifier  # 决策树

y_score = model_train(DecisionTreeClassifier())
fpr, tpr, auc_val = roc_metric(y_test, y_score)
plt_ROC(fpr, tpr, auc_val, 'DecisionTreeClassifier')

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # 线性判别分析

y_score = model_train(LinearDiscriminantAnalysis())
fpr, tpr, auc_val = roc_metric(y_test, y_score)
plt_ROC(fpr, tpr, auc_val, 'LinearDiscriminantAnalysis')

from sklearn.neighbors import KNeighborsClassifier  # k近邻

y_score = model_train(KNeighborsClassifier())
fpr, tpr, auc_val = roc_metric(y_test, y_score)
plt_ROC(fpr, tpr, auc_val, 'KNeighborsClassifier')
plt.plot([0,1],[0,1], color='navy', lw=1, ls='--')
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.grid()
plt.title('Multiclass Classification ')
plt.legend(fontsize=12)
plt.show()
