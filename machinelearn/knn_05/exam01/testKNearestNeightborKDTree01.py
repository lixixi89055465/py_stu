# -*- coding: utf-8 -*-
# @Time : 2024/2/4 17:53
# @Author : nanji
# @Site : 
# @File : testKNearestNeightborKDTree01.py
# @Software: PyCharm 
# @Comment : 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from machinelearn.knn_05.knn_kdtree import KNearestNeighborKDTree
from machinelearn.knn_05.exam01.KNearestNeighbor_KDTree import KNearestNeighborKDTree1
# from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
knn = KNearestNeighborKDTree1(k=3, p=2, view_kdt=True)
knn.fit(X_train, y_train)
y_test_pred = knn.predict(X_test)
print(classification_report(y_test, y_test_pred))
print('1'*100)
# knn = KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree')
# knn.fit(X_train, y_train)
# y_test_pred = knn.predict(X_test)
# print(classification_report(y_test, y_test_pred))
# print('2'*100)


