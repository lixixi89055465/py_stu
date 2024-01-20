# -*- coding: utf-8 -*-
# @Time : 2024/1/19 11:07
# @Author : nanji
# @Site : 
# @File : 10_3.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l
import math


def masked_softmax(X, valid_lens):
	if valid_lens is None:
		return nn.functional.softmax(X, dim=-1)
	else:
		shape = X.shape
		if valid_lens.dim() == 1:
			valid_lens = torch.repeat_interleave(valid_lens, shape[1])
		else:
			valid_lens = valid_lens.reshape(-1)
		X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, \
							  value=-1e6)
		return nn.functional.softmax(X.reshape(shape), dim=-1)


# result=masked_softmax(torch.rand(2,2,4),torch.tensor([2,3]))
# print(result)
result = masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 3]]))
print(result)
