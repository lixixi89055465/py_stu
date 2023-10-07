# -*- coding: utf-8 -*-
# @Time    : 2023/10/5 10:09
# @Author  : nanji
# @Site    : 
# @File    : test_svm_classifier.py
# @Software: PyCharm 
# @Comment :
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from machinelearn.svm_06.svm_smo_classifier import SVMClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=200, n_features=2, n_classes=2, n_informative=1,
                           n_redundant=0, n_repeated=0, n_clusters_per_class=1,
                           class_sep=1.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=y)
svm = SVMClassifier(C=1000)
svm.fit(X_train, y_train)
y_test_pred = svm.predict(X_test)
print(classification_report(y_test, y_test_pred))

plt.figure(figsize=(14, 5))
plt.subplot(121)
svm.plt_svm(X_train, y_train, is_show=False,is_margin=True)
plt.subplot(122)
svm.plt_loss_curve(is_show=False)
plt.show()
