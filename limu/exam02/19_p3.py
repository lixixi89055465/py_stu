# -*- coding: utf-8 -*-
# @Time    : 2023/11/25 上午10:04
# @Author  : nanji
# @Site    : 
# @File    : 19_p3.py
# @Software: PyCharm 
# @Comment :
import torch
from torch import nn
from d2l import torch as d2l


def corr2d(X, K):
    '''
    计算二维相关互运算。
    :param X:
    :param K:
    :return:
    '''
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


X = torch.tensor([[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]])
K = torch.tensor([[0., 1.], [2., 3.]])
print('0' * 100)
print(corr2d(X, K))


class Conv2d(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        return corr2d(X, self.weight) + self.bias


X = torch.ones((6, 8))
X[:, 2:6] = 0
# print('1' * 100)
# print(X)
K = torch.tensor([[1.0, -1.]])

print('2' * 100)
Y = corr2d(X, K)
# print(Y)
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'batch {i + 1}, loss {l.sum():.3f}')

print('3'*100)
print(conv2d.weight.data.shape)
print('4'*100)
print(conv2d.weight.data.reshape((1, 2)))