# -*- coding: utf-8 -*-
# @Time : 2024/1/3 16:43
# @Author : nanji
# @Site : 
# @File : testReduction.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l
import torch.nn.functional as F

out = torch.Tensor([[1, 2, 3], [3, 4, 1]])
target = torch.LongTensor([0, 1])

loss = F.cross_entropy(out, target)
print(loss)

loss = F.cross_entropy(out, target, reduction='sum')
print('0' * 100)
print(loss)
print('1'*100)
loss = F.cross_entropy(out, target, reduction='none')
print(loss)
