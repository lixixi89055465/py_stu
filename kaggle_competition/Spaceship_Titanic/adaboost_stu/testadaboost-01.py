# -*- coding: utf-8 -*-
# @Time : 2024/1/7 16:20
# @Author : nanji
# @Site : https://www.bilibili.com/video/BV1Ca4y1t7DS?p=7&vd_source=50305204d8a1be81f31d861b12d4d5cf1
# @File : testadaboost-01.py
# @Software: PyCharm 
# @Comment :

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

x1, y1 = make_gaussian_quantiles(n_samples=1000, cov=2., \
								 n_features=2, n_classes=2, \
								 shuffle=True, random_state=1)

x2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5, \
								 n_samples=300, n_features=2, \
								 n_classes=2, shuffle=True, random_state=1)

# 合并
X = np.vstack((x1, x2))
y = np.hstack((y1, 1 - y2))
# 绘制生成数据
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()
# 设定弱分类器
weakClassifier = DecisionTreeClassifier(max_depth=2)
# 构建模型并进行训练
clf = AdaBoostClassifier(base_estimator=weakClassifier, \
						 algorithm='SAMME', \
						 n_estimators=300, \
						 learning_rate=0.8)
clf.fit(X, y)
# 模型预测
y_ = clf.predict(np.c_[x1.ravel(), x2.ravel()])

print(y_.shape)