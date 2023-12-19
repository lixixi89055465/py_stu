# -*- coding: utf-8 -*-
# @Time    : 2023/12/19 23:20
# @Author  : nanji
# @Site    : 
# @File    : 55_p1.py
# @Software: PyCharm 
# @Comment :
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
