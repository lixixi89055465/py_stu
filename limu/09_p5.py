# -*- coding: utf-8 -*-
# @Time    : 2023/11/17 下午11:13
# @Author  : nanji
# @Site    : 
# @File    : 09_p5.py
# @Software: PyCharm 
# @Comment :
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
print(batch_size)

# pytorch 不会隐式地调整输入的形状。
# 因此，我们定义了展平层（flatten)在现行层前调整网络输入
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 10
d2l.train_epoch_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
