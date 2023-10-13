# -*- coding: utf-8 -*-
# @Time    : 2023/10/9 下午9:14
# @Author  : nanji
# @Site    : 
# @File    : test_nb_classifier.py
# @Software: PyCharm 
# @Comment :

import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs, load_iris
from machinelearn.bayesian_07.lecture.naive_bayes_classifier import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from machinelearn.bayesian_07.utils.plt_decision_function import plot_decision_function
import matplotlib.pyplot as plt

# wm = pd.read_csv("../datasets/watermelon.csv").dropna()
# X, y = np.asarray(wm.iloc[:, 1:-1]), np.asarray(wm.iloc[:, -1])
# # nbc = NaiveBayesClassifier(is_binned=True, feature_R_idx=[6, 7])
# # nbc = NaiveBayesClassifier(is_binned=False, feature_R_idx=[6, 7])
# nbc = NaiveBayesClassifier(is_binned=False, feature_R_idx=[6, 7],max_bins=10)
# nbc.fit(X, y)
#
# y_proba = nbc.predict_proba(X)
# print(y_proba)
# print('0'*100)
# y_hat = nbc.predict(X)
# print(y_hat)
# X, y = make_blobs(n_features=2, n_samples=500, centers=4, cluster_std=0.85, random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, \
#                                                     random_state=0, stratify=y)

# nbc = NaiveBayesClassifier(is_binned=True, max_bins=20, is_feature_all_R=True)
# nbc.fit(X_train, y_train)
# y_pred = nbc.predict(X_test)
# print('0' * 100)
# print(classification_report(y_test, y_pred))
# plot_decision_function(X_train, y_train, nbc)
# plt.figure(figsize=(14, 5))
# plt.subplot(121)
# plot_decision_function(X_train, y_train, nbc, is_show=False)

# nbc = NaiveBayesClassifier(is_binned=False, feature_R_idx=[0, 1])
# nbc.fit(X_train, y_train)
# y_pred = nbc.predict(X_test)
# print(classification_report(y_test, y_pred))
# plt.subplot(121)
# plot_decision_function(X_train, y_train, nbc, is_show=False)
# plt.show()

# iris=load_iris()
# X,y=iris.data,iris.target
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,\
#                                                random_state=0,stratify=y)
# nbc=NaiveBayesClassifier(is_binned=True,max_bins=15,is_feature_all_R=True)
# nbc.fit(X_train,y_train)
# y_pred=nbc.predict(X_test)
# print(classification_report(y_test,y_pred ))
from sklearn.preprocessing import LabelEncoder

# al = pd.read_csv('../datasets/agaricus-lepiota.data').dropna()
# X, y = np.asarray(al.iloc[:, 1:]), np.asarray(al.iloc[:, 0])
# y = LabelEncoder().fit_transform(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
# nbc = NaiveBayesClassifier(is_binned=False)
# nbc.fit(X_train, y_train)
# y_pred = nbc.predict(X_test)
# print(classification_report(y_test, y_pred))



from sklearn.datasets import load_digits
digits=load_digits()
X,y= digits.data, digits.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,\
                                               random_state=0,stratify=y)
nbc=NaiveBayesClassifier(is_binned=True,max_bins=30,is_feature_all_R=True)
nbc.fit(X_train,y_train)
y_pred=nbc.predict(X_test)
print(classification_report(y_test,y_pred ))