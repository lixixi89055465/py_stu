# -*- coding: utf-8 -*-
# @Time : 2024/1/5 23:45
# @Author : nanji
# @Site : https://www.cnblogs.com/hider/p/15978785.html
# @File : testmake_blobs.py
# @Software: PyCharm 
# @Comment :


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

X, y = make_blobs(n_samples=1000, \
				  n_features=2,
				  centers=2, \
				  cluster_std=1.5, \
				  random_state=1)
# plt.style.use('ggplot')
# plt.figure()
# plt.title('Data')
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=np.squeeze(y), s=30)
# plt.show()
X, y = make_blobs(n_samples=[100, 300, 250, 400],
				  n_features=2, \
				  centers=[[100, 150], [250, 400], [600, 100], [300, 500]], \
				  cluster_std=50, \
				  random_state=1)
plt.style.use('ggplot')
plt.figure()
plt.title('Data')
plt.scatter(X[:, 0], X[:, 1], marker='o', c=np.squeeze(y), s=30)
plt.show()
