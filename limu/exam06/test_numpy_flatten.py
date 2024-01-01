# -*- coding: utf-8 -*-
# @Time : 2024/1/1 17:01
# @Author : nanji
# @Site : 
# @File : test_numpy_flatten.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l
import numpy as np
import torch

x = np.arange(27)
x = np.reshape(x, (3, 3, 3))

x = torch.from_numpy(x)
x = torch.flatten(x)

x = np.arange(27)
x = np.reshape(x, (3, 3, 3))
x = torch.from_numpy(x)
x = torch.flatten(x, start_dim=0, end_dim=1)
# print('after flatten ', x)
print('0'*100)

