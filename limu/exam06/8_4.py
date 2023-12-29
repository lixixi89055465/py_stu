# -*- coding: utf-8 -*-
# @Time : 2023/12/29 13:44
# @Author : nanji
# @Site : 
# @File : 8_4.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l

X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))

# 3,4 + 3,4
a = torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
print(a)
