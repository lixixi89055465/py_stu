# -*- coding: utf-8 -*-
# @Time : 2023/12/30 11:32
# @Author : nanji
# @Site : 
# @File : 67_p2.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l

encoding_dim, num_steps = 32, 60
pos_encoding = d2l.PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(torch.arange(num_steps), \
		 P[0:, 6:10].T, xlabel='Row', \
		 figsize=(6, 2.5), \
		 legend=['Col %d' % d for d in torch.arange(6, 10)])
