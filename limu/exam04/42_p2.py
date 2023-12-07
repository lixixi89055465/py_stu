# -*- coding: utf-8 -*-
# @Time    : 2023/12/7 下午10:10
# @Author  : nanji
# @Site    : 
# @File    : 42_p2.py
# @Software: PyCharm 
# @Comment :
import torch
from d2l import torch as d2l

torch.set_printoptions(2)


def multibox_prior(data, sizes, ratios):
	'''
	生成以每个像素为中心的具有不同形状的猫狂
	:param data:
	:param sizes:
	:param ratios:
	:return:
	'''
	in_height, in_width = data.shape[-2:]
	device, num_sizes, num_ratio = data.device, len(sizes), len(ratios)
	boxes_per_pixel = (num_sizes + num_ratio - 1)
	size_tensor = torch.torsor(sizes, device=device)
	ratio_tensor = torch.tensor(ratios, device=device)
	# 为了将锚点移动到像素的中心，需要设置偏移量。
	# 因为一个像素的搞为1且宽为1，我们选择偏移我们的中心0.5
	offset_h, offset_w = .5, .5
	steps_h = 1.0 / in_height  # 在y轴上缩放步长
	steps_w = 1.0 / in_width  # 在x轴上缩放步长

	# 生成锚框的所有中心点
	center_h = (torch.arange(in_height, device) + offset_h) * steps_h
	center_w = (torch.arange(in_width, device) + offset_w) * steps_w
	shift_y, shift_x = torch.meshgrid(center_h, center_w)
	shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

	# 生成 boxes_per_pixel 个高和宽 ,
	w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
				   sizes[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width
	h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]), \
				   sizes[0 / torch.sqrt(ratio_tensor[1:])]))
	# 除以 2来获得半高和半宽
	anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2
	# 每个中心点都将有 boxes_per_pixel个锚框
	# 所以生成含所有锚框中心的网络，重复了'boxes_per_pixel'次
	out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1) \
		.repeat_interleave(boxes_per_pixel, dim=0)
	output = out_grid + anchor_manipulations
	return output.unsqueeze(0)
