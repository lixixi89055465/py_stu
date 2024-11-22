# -*- coding: utf-8 -*-
# @Time    : 2024/11/17 下午10:58
# @Author  : nanji
# @Site    : https://blog.csdn.net/bigFatCat_Tom/article/details/91619977
# @File    : testBatchNorm2d.py
# @Software: PyCharm 
# @Comment :
# encoding:utf-8
import torch
import torch.nn as nn

# num_features - num_features from an expected input of size:batch_size*num_features*height*width
# eps:default:1e-5 (公式中为数值稳定性加到分母上的值)
# momentum:动量参数，用于running_mean and running_var计算的值，default：0.1
m = nn.BatchNorm2d(2, affine=True)  # affine参数设为True表示weight和bias将被使用
input = torch.randn(1, 2, 3, 4)
output = m(input)

print(input)
print(m.weight)
print(m.bias)
print(output)
print(output.size())