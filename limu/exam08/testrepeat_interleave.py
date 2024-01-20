# -*- coding: utf-8 -*-
# @Time : 2024/1/18 9:48
# @Author : nanji
# @Site : 
# @File : testrepeat_interleave.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l
y = torch.tensor([[1, 2], [3, 4]])
result=torch.repeat_interleave(y, torch.tensor([1, 2]), dim=0)
print(result)
result=torch.repeat_interleave(y, torch.tensor([1, 2]), dim=0, output_size=3)
print('0'*100)
print(result)
