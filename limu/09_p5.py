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

batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
print(batch_size)
