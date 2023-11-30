# -*- coding: utf-8 -*-
# @Time    : 2023/11/29 下午11:08
# @Author  : nanji
# @Site    : 
# @File    : 26_p2.py
# @Software: PyCharm 
# @Comment :
import torch
from torch import nn
from d2l import torch as d2l

def nin_block(in_channels,out_channels,kernel_size,strides,padding):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,strides,padding),
        nn.ReLU(),nn.Conv2d(out_channels,out_channels,kernel_size=1),
        nn.ReLU(), nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
    )

net=nn.Sequential(
    nin_block(1,96,kernel_size=11,strides=4,padding=0),
    nn.MaxPool2d(3,stride=2),
    nin_block(96,256,kernel_size=5,strides=1,padding=2),
    nn.MaxPool2d(3,stride=2),
    nin_block(256,384,kernel_size=3,strides=1,padding=1),
    nn.MaxPool2d(3,stride=2,),nn.Dropout(p=0.5),
    nin_block(385,10,kernel_size=3,strides=1,padding=1),
    nn.AdaptiveAvgPool2d((1,1)) ,
    nn.Flatten(),
)

