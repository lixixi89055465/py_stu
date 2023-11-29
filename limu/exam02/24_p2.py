# -*- coding: utf-8 -*-
# @Time    : 2023/11/28 下午10:05
# @Author  : nanji
# @Site    : 
# @File    : 24_p2.py
# @Software: PyCharm 
# @Comment :
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    # 1*224*224=>96*54*54
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 96*26*26
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    # 256*26*26
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 256*12*12
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    # 384*12*12
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    # 384*12*12
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    # 256*12*12
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 256*5*5
    nn.Flatten(),
    # 6400
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 10)
)

X = torch.randn(1, 1, 224, 224)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'Output shape:\t', X.shape)

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
plt.savefig('1.png')
