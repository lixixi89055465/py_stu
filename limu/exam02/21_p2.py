# -*- coding: utf-8 -*-
# @Time    : 2023/11/25 下午8:51
# @Author  : nanji
# @Site    : 
# @File    : 21_p2.py
# @Software: PyCharm 
# @Comment :
import torch
from d2l import torch as d2l


def corr2d_multi_in(X, K):
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))


X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6., 7., 8.]],
                  [[1.0, 2.0, 3.0], [4., 5., 6.], [7., 8., 9.]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.]],
                  [[1., 2.], [3., 4.]]])

print(X.shape)
print(K.shape)
a = corr2d_multi_in(X, K)
print(a)
print(a.shape)
print('0' * 100)


def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K])


print('1' * 100)
K = torch.stack((K, K + 1, K + 2), dim=0)
b = corr2d_multi_in_out(X, K)
print(b)
print('2' * 100)


def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))
X=torch.normal(0,1,(3,3,3))
K=torch.normal(0,1,(2,3,1,1))
Y1=corr2d_multi_in_out_1x1(X,K)
Y2=corr2d_multi_in_out(X,K)
print('3'*100)
assert float(torch.abs(Y1-Y2).sum())<1e-6
