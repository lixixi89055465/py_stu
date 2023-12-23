# -*- coding: utf-8 -*-
# @Time    : 2023/12/23 22:52
# @Author  : nanji
# @Site    : 
# @File    : testnn_embedding.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l

embedding = nn.Embedding(10, 3)
input = torch.LongTensor([
    [1, 2, 4, 5],
    [4, 3, 2, 9]
])
e = embedding(input)
print(embedding.weight)
print(e)
