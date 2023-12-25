# -*- coding: utf-8 -*-
# @Time    : 2023/12/24 11:17
# @Author  : nanji
# @Site    : 
# @File    : testMask.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l

X = torch.ones((2, 3))
print(X[None, :])
maxlen = X.size(1)
torch.arange((maxlen))
valid_len = torch.tensor([1, 2])
print(torch.arange((maxlen), dtype=torch.float32, \
                   device=X.device)[None, :] < valid_len[:, None])

print('1' * 100)
# print(torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :]<valid_len[:,None])
print(valid_len[:, None].shape)
print(torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :].shape)
print(torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None])
print('2' * 100)
print(valid_len[:, None])
print(torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None])
