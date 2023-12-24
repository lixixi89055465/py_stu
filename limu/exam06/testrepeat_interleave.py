# -*- coding : utf-8 -*-
# @Time     : 2023/12/24 21:09
# @Author   : nanji
# @Site     :
# @File     : testrepeat_interleave.py
# @Software : PyCharm
# @Comment  :
import os
import torch
from torch import nn
from d2l import torch as d2l

x = torch.tensor([1, 2, 3])
print(x.repeat_interleave(2))
# 传入多维张量 ，默认`展平`
y = torch.tensor([[1, 2], [3, 4]])
torch.rprint(x.repeat_interleave(2))
# 指定维度
torch.repeprint(x.repeat_interleave(2))
print('0'*100)
torch.repeat_iprint(x.repeat_interleave(2))
# 指定不同元素重复不同次数
print(torch.repeat_interleave(y, torch.tensor([1, 2]), dim=0))
