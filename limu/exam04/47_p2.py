# -*- coding: utf-8 -*-
# @Time    : 2023/12/10 上午10:30
# @Author  : nanji
# @Site    : 
# @File    : 47_p2.py
# @Software: PyCharm 
# @Comment :
import torch
from torch import nn
from d2l import torch as d2l


def trans_conv(X, K):
	h, w = K.shape
	Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			Y[i:i + h, j:j + w] += X[i, j] * K
	return Y


X = torch.tensor([[0.0, 1.0], [2.,  3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print('0' * 100)
print(trans_conv(X, K))
