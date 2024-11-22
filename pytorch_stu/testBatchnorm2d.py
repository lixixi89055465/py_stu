# -*- coding: utf-8 -*-
# @Time    : 2024/11/18 下午9:17
# @Author  : nanji
# @Site    : 
# @File    : testBatchnorm2d.py
# @Software: PyCharm 
# @Comment :
# https://blog.csdn.net/algorithmPro/article/details/103982466
# 在训练阶段
import torch.nn as nn
import torch
import copy

m3 = nn.BatchNorm2d(3, eps=0, momentum=0.5, affine=True, track_running_stats=True).cuda()
# 为了方便验证，设置模型参数的值
m3.running_mean = (torch.ones([3]) * 4).cuda()  # 设置模型的均值是4
m3.running_var = (torch.ones([3]) * 2).cuda()  # 设置模型的方差是2

# 查看模型参数的值
print('trainning:', m3.training)
print('running_mean:', m3.running_mean)
print('running_var:', m3.running_var)
# gamma对应模型的weight，默认值是1
print('weight:', m3.weight)
# gamma对应模型的bias，默认值是0
print('bias:', m3.bias)

ex_old = copy.deepcopy(m3.running_mean)
var_old = copy.deepcopy(m3.running_var)
# 计算更新后的均值和方差
momentum = m3.momentum  # 更新参数
# >
# trainning: True
# running_mean: tensor([4., 4., 4.], device='cuda:0')
# running_var: tensor([2., 2., 2.], device='cuda:0')
# weight: Parameter
# containing:
# tensor([1., 1., 1.], device='cuda:0', requires_grad=True)
# bias: Parameter
# containing:
# tensor([0., 0., 0.], device='cuda:0', requires_grad=True)

# 生成通道3，416行416列的输入数据
torch.manual_seed(21)
input3 = torch.randn(2, 3, 416, 416).cuda()
# 输出第一个通道的数据
# input3[0][0]
# 数据归一化
output3 = m3(input3)
# 输出归一化后的第一个通道的数据
# output3[0][0]
print('*' * 30)
print('程序计算的新的均值ex_new:', m3.running_mean)
print('程序计算的新的方差var_new:', m3.running_var)
print("程序计算的输出bn：")
print(output3[0])

# 输入数据的均值
# input3[0][i].mean()单个样本单个通道的均值
# (input3[0][i].mean()+input3[1][i].mean())/2 所有样本单个通道的均值（但是这里只有2个样本）
obser_mean = torch.Tensor([(input3[0][i].mean() + input3[1][i].mean()) / 2 for i in range(3)]).cuda()
# 输入数据的方差
obser_var = torch.Tensor([(input3[0][i].var() + input3[1][i].var()) / 2 for i in range(3)]).cuda()

# 更新均值
ex_new = (1 - momentum) * ex_old + momentum * obser_mean
# 更新方差
var_new = (1 - momentum) * var_old + momentum * obser_var
# 打印
print('*' * 30)
print('手动计算的新的均值ex_new:', ex_new)
print('手动计算的新的方差var_new:', var_new)

# # >
# ex_new: tensor([2.0024, 2.0015, 2.0007], device='cuda:0')

# var_new: tensor([1.5024, 1.4949, 1.5012], device='cuda:0')

output3_calcu = torch.zeros_like(input3)
for channel in range(input3.shape[1]):
	output3_calcu[0][channel] = (input3[0][channel] - obser_mean[channel]) / (pow(obser_var[channel] + m3.eps, 0.5))
# 编码归一化
# output3_channel_1 = (input3[0][0] - obser_mean[0]) / (pow(obser_var[0] + m3.eps, 0.5))
# output3_channel_2 = (input3[0][1] - obser_mean[1]) / (pow(obser_var[1] + m3.eps, 0.5))
# output3_channel_3 = (input3[0][2] - obser_mean[2]) / (pow(obser_var[2] + m3.eps, 0.5))
# output3_source
print("手动计算的输出bn：")
print(output3_calcu[0])