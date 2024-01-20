# -*- coding: utf-8 -*-
# @Time : 2024/1/16 10:47
# @Author : nanji
# @Site : 
# @File : 10_1.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l

def show_heatmaps(matrices,xlabel,ylabel,titles=None,\
				  figsize=(2.5,2.5),\
				  cmap='Reds'):
	d2l.use_svg_display()
	num_rows,num_cols=matrices.shape[0],matrices.shape[1]
	fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,\
								 sharex=True, sharey=True, squeeze=False)
	for i,(row_axes,row_matrices) in enumerate(zip(axes,matrices)):
		for j,(ax,matrix) in enumerate(zip(row_axes,row_matrices)):
			pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
			if i == num_rows - 1:
				ax.set_xlabel(xlabel)
			if j == 0:
				ax.set_ylabel(ylabel)
			if titles:
				ax.set_title(titles[j])
		fig.colorbar(pcm, ax=axes, shrink=0.6)

attention_weight=torch.eye(10).reshape((1,1,10,10))
show_heatmaps(attention_weight,xlabel='keys',ylabel='Queries')

