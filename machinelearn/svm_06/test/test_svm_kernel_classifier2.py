# -*- coding: utf-8 -*-
# @Time    : 2023/10/5 10:09
# @Author  : nanji
# @Site    : 
# @File    : test_svm_classifier.py
# @Software: PyCharm 
# @Comment :
import numpy as np
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from machinelearn.svm_06.svm_smo_classifier import SVMClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# X, y = make_moons(n_samples=200, noise=0.3)
X, y = make_classification(n_samples=200, n_features=2, n_classes=2, n_informative=1,
                           n_redundant=0, n_repeated=0, n_clusters_per_class=1,
                           class_sep=1.5, random_state=21)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=y)

svm_rbf1 = SVMClassifier(C=10.0, kernel='rbf', gamma=0.2)
svm_rbf1.fit(X_train, y_train)
y_test_pred = svm_rbf1.predict(X_test)
print(classification_report(y_test, y_test_pred))
print('='*100)

svm_rbf2 = SVMClassifier(C=10.0, kernel='rbf', gamma=0.2)
svm_rbf2.fit(X_train, y_train)
y_test_pred = svm_rbf2.predict(X_test)
print(classification_report(y_test, y_test_pred))
print('='*100)


svm_poly1 = SVMClassifier(C=10.0, kernel='poly', degree=3)
svm_poly1.fit(X_train, y_train)
y_test_pred = svm_poly1.predict(X_test)
print(classification_report(y_test, y_test_pred))

svm_poly2 = SVMClassifier(C=10.0, kernel='poly', degree=6)
svm_poly2.fit(X_train, y_train)
y_test_pred = svm_poly2.predict(X_test)
print(classification_report(y_test, y_test_pred))




X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_informative=1,
                           n_redundant=0, n_repeated=0, n_clusters_per_class=1,
                           class_sep=1.5, random_state=21)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=y)

svm_linear1= SVMClassifier(C=10.0, kernel='linear')
svm_linear1.fit(X_train1, y_train1)
y_test_pred1 = svm_linear1.predict(X_test1)
print(classification_report(y_test1, y_test_pred1))

svm_linear2 = SVMClassifier(C=10.0, kernel='linear')
svm_linear2.fit(X_train1, y_train1)
y_test_pred1 = svm_linear2.predict(X_test1)
print(classification_report(y_test1, y_test_pred1))




plt.figure(figsize=(21, 10))
plt.subplot(231)
svm_rbf1.plt_svm(X_train, y_train, is_show=False)
plt.subplot(232)
svm_rbf2.plt_svm(X_train, y_train, is_show=False)

plt.subplot(233)
svm_poly1.plt_svm(X_train, y_train, is_show=False)
plt.subplot(234)
svm_poly2.plt_svm(X_train, y_train, is_show=False)


plt.subplot(235)
svm_linear1.plt_svm(X_train1, y_train1, is_show=False,is_margin=True)
plt.subplot(236)
svm_linear2.plt_svm(X_train1, y_train1, is_show=False,is_margin=True)


plt.show()
