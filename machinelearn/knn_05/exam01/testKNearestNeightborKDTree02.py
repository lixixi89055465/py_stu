# -*- coding: utf-8 -*-
# @Time : 2024/2/4 17:54
# @Author : nanji
# @Site : 
# @File : testKNearestNeightborKDTree02.py
# @Software: PyCharm 
# @Comment :
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from machinelearn.knn_05.exam01.KNearestNeighbor_KDTree import KNearestNeighborKDTree1
import numpy as np
from sklearn.metrics import accuracy_score

bc_data = load_breast_cancer()
X, y = bc_data.data, bc_data.target
X = StandardScaler().fit_transform(X)
test_accuracy_scores = []
train_accuracy_scores = []
k_neighbors = np.arange(1, 21)
acc = []
for k in k_neighbors:
	test_scores, train_scores = [], []
	k_fold = StratifiedKFold(n_splits=10).split(X, y)
	for train_idx, test_idx in k_fold:
		knn = KNearestNeighborKDTree1(k=k, p=2)
		knn.fit(X[train_idx], y[train_idx])
		y_train_pred = knn.predict(X[train_idx])
		y_test_pred = knn.predict(X[test_idx])
		test_scores.append(accuracy_score(y[test_idx],y_test_pred))
		train_scores.append(accuracy_score(y[train_idx],y_train_pred))
		del knn

