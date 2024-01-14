# -*- coding: utf-8 -*-
# @Time : 2024/1/14 14:42
# @Author : nanji
# @Site : https://www.bilibili.com/video/BV1Ca4y1t7DS?p=7&vd_source=50305204d8a1be81f31d861b12d4d5cf
# @File : testAdaboost.py
# @Software: PyCharm 
# @Comment :


import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

x1, y1 = make_gaussian_quantiles(
	cov=2, n_samples=200, n_features=2, \
	n_classes=2, shuffle=True, random_state=1
)
x2, y2 = make_gaussian_quantiles(
	mean=(3, 3), cov=1.5, n_samples=300, \
	n_features=2, n_classes=2, shuffle=True, random_state=1
)
X = np.vstack((x1, x2))
y = np.hstack((y1, 1 - y2))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

print('0' * 100)
weakClassifier = DecisionTreeClassifier(max_depth=2)
clf = AdaBoostClassifier(base_estimator=weakClassifier, \
						 algorithm='SAMME', \
						 n_estimators=300, \
						 learning_rate=0.8, )
clf.fit(X, y)

a=np.c_[x1.ravel(), x2.ravel()]
# y_ = clf.predict(np.c_[x1.ravel(), x2.ravel()])
print(a.shape)
