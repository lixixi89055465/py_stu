# -*- coding: utf-8 -*-
# @Time : 2024/1/18 18:01
# @Author : nanji
# @Site : 
# @File : testtensorrepeat.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l

x = torch.tensor([1, 2, 3])
print(x.shape)
print('0'*100)
print(x.repeat(4, 2))