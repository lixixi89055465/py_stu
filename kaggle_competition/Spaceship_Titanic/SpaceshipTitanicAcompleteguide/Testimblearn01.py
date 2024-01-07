# -*- coding: utf-8 -*-
# @Time : 2024/1/3 22:10
# @Author : nanji
# @Site : https://zhuanlan.zhihu.com/p/95020088
# @File : Testimblearn01.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(n_samples=200, n_features=2, n_classes=2, n_informative=2,
						   n_redundant=0, n_clusters_per_class=1,
						   class_sep=0.8, random_state=21)
num_range = np.arange(1, 100)
num_0 = np.random.choice(num_range, size=50, replace=False)

X0 = X[y == 0]
y0 = y[y == 0]
X1 = X[y == 1][num_0]
y1 = y[y == 1][num_0]
X = np.concatenate([X0, X1], axis=0)
y = np.concatenate([y0, y1], axis=0)
X_shuffle_index = np.arange(0, 150)
np.random.shuffle(X_shuffle_index)
X=X[X_shuffle_index]
y=y[X_shuffle_index]
rus = RandomUnderSampler(random_state=0)

X_resampled, y_resampled = rus.fit_resample(X, y)
print(sorted(Counter(y).items()))
print(sorted(Counter(y_resampled).items()))
