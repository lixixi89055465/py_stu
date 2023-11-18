# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/11/18 20:25
# @Author  : nanji
# @File    : 11_p3.py
# @Description :https://www.bilibili.com/video/BV1kX4y1g7jp/?p=3&spm_id_from=pageDriver&vd_source=50305204d8a1be81f31d861b12d4d5cf


import torch
import numpy as np
import math
from torch import nn
from d2l import torch as d2l

max_degree = 20
n_train, n_test = 100, 100
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)
true_w, features, poly_features, labels = \
    [torch.tensor(x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]]
print('0' * 100)

print(features[:2], poly_features[:2, :], labels[:2])
