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


X = torch.tensor([[0.0, 1.0], [2., 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print('0' * 100)
print(trans_conv(X, K))

X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
print('1' * 100)
print(tconv(X))
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
print('2' * 100)
print(tconv(X))

tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=1, bias=False)
tconv.weight.data = K
print('3' * 100)
print(tconv(X))

X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2)
print('4' * 100)
print(tconv(conv(X)).shape == X.shape)

# 与矩阵变换的关系
X = torch.arange(9.0).reshape(3, 3)
K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X, K)
print('5' * 100)
print(Y)


def kernel2matrix(K):
	k, W = torch.zeros(5), torch.zeros((4, 9))
	k[:2], k[3:5] = K[0, :], K[1, :]
	W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
	return W


W = kernel2matrix(K)
print('6' * 100)
print(W)
print(Y == torch.matmul(W, X.reshape(-1)).reshape(2, 2))
Z = trans_conv(Y, K)
print('7' * 100)
print(Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3))
