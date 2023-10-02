# -*- coding: utf-8 -*-
# @Time    : 2023/10/2 20:53
# @Author  : nanji
# @Site    : 
# @File    : test_knn_1.py
# @Software: PyCharm 
# @Comment :
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from machinelearn.knn_05.knn_kdtree import KNearestNeighborKDTree
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# iris = load_iris()
iris = load_breast_cancer()
X, y = iris.data, iris.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                    test_size=0.3, random_state=0, stratify=y)

k_neighbors = np.arange(3, 21)
acc = []
from sklearn.metrics import accuracy_score

for k in k_neighbors:
    knn = KNearestNeighborKDTree(k=k)
    knn.fit(X_train, y_train)
    y_test_hat = knn.predict(X_test)
    a = accuracy_score(y_test, y_test_hat)
    acc.append(a)
    print("k = %d\t %.2f" % (k, a))
# print(classification_report(y_test, y_test_hat))
plt.figure(figsize=(7, 5))
plt.plot(k_neighbors, acc, 'ko-', lw=1.)
plt.show()
