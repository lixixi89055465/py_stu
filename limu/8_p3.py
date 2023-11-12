# -*- coding: utf-8 -*-
# @Time    : 2023/11/12 22:04
# @Author  : nanji
# @Site    : 
# @File    : 8_p3.py
# @Software: PyCharm 
# @Comment :
import torch
import numpy as np
import matplotlib.pyplot as plt

# from d2l.mxnet import synthetic_data
print(torch.__version__)

true_w = torch.tensor([2, -3.4])
true_b = 4.2


def synthetic_data(true_w, true_b, num_examples):
    '''生成y=Xw+b + 噪声 '''
    X = torch.normal(mean=0, std=1, size=(num_examples, len(true_w)))
    y = torch.matmul(X, true_w) + true_b
    y += torch.normal(mean=0, std=0.01, size=y.shape)
    return X, y


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i:min(i + batch_size, num_examples)]
        yield features[batch_indices], labels[batch_indices]


batch_size = 10
features, labels = synthetic_data(true_w, true_b, 1000)
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.ones(1, requires_grad=True)


def linreg(X, w, b):
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    return ((y_hat - y.reshape(y_hat.shape)) ** 2) / 2


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


lr = 0.03
num_epoch = 3
net = linreg
loss = squared_loss
for epoch in range(num_epoch):
    for X, y in data_iter(batch_size, features, labels):
        Loss = loss(net(X, w, b), y)
        Loss.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1} , loss={float(train_l.mean()):f}')

print('0'*100)

print(true_w)
print(w)
print(true_b)
print(b)
