# -*- coding: utf-8 -*-
# @Time    : 2023/9/23 上午8:32
# @Author  : nanji
# @Site    : 
# @File    : test_lda2classify.py
# @Software: PyCharm 
# @Comment :
import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer
from machinelearn.linear_model_03.lda_lecture.lda2classify import LDABinaryClassifier

iris = load_iris()
X, y = iris.data[:100, :], iris.target[:100]
# bc_data = load_breast_cancer()
# X, y = bc_data.data, bc_data.target
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22, stratify=y)
lda = LDABinaryClassifier()
lda.fit(X_train, y_train)
y_test_pred=lda.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_test_pred))
