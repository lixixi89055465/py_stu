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

X, y = make_moons(n_samples=200, noise=0.3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=y)

svm_rbf = SVMClassifier(C=10.0, kernel='rbf', gamma=0.2)
svm_rbf.fit(X_train, y_train)
y_test_pred = svm_rbf.predict(X_test)
print(classification_report(y_test, y_test_pred))
print('='*100)

svm_poly = SVMClassifier(C=10.0, kernel='poly', degree=5)
svm_poly.fit(X_train, y_train)
y_test_pred = svm_poly.predict(X_test)
print(classification_report(y_test, y_test_pred))


plt.figure(figsize=(14, 10))
plt.subplot(221)
svm_rbf.plt_svm(X_train, y_train, is_show=False)
plt.subplot(222)
svm_rbf.plt_loss_curve(is_show=False)

plt.subplot(223)
svm_poly.plt_svm(X_train, y_train, is_show=False)
plt.subplot(224)
svm_poly.plt_loss_curve(is_show=False)


plt.show()
