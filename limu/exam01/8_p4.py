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
import numpy as np
from torch.utils import data


def synthetic_data(true_w, true_b, num_examples):
    '''生成y=Xw+b + 噪声 '''
    X = torch.normal(mean=0, std=1, size=(num_examples, len(true_w)))
    y = torch.matmul(X, true_w) + true_b
    y += torch.normal(mean=0, std=0.01, size=y.shape)
    return X, y.reshape(-1,1)


def load_array(data_arrays, batch_size, is_train=True):
    '''
    构造一个pytorch数据迭代器
    :param data_arrays:
    :param batch_size:
    :param is_train:
    :return:
    '''
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
batch_size = 10
data_iter = load_array((features, labels), batch_size)
print('0' * 100)

# print(next(iter(data_iter)))
from torch import nn

net = nn.Sequential(nn.Linear(in_features=2, out_features=1))
net[0].weight.data.normal_(0, 0.1)
net[0].bias.data.fill_(0)
print('1' * 100)
print(net)
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
print('2' * 100)

print(trainer)
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1} , loss {l:f}')
