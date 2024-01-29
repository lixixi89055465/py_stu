# -*- coding: utf-8 -*-
# @Time : 2024/1/27 20:42
# @Author : nanji
# @Site : 
# @File : DataBinWrapper.py
# @Software: PyCharm 
# @Comment : 决策树自编算法——连续数值型数据分箱类DataBinWrapper

import numpy as np


class DataBinWrapper:
	'''针对连续数据进行分组（分箱）操作 '''

	def __init__(self, max_bins=0):
		self.max_bins = max_bins
		self.XrangeMap = None  #

	def fit(self, x):
		'''
		数据的拟合与处理，对于给定的数据按照分段数计算分位数，
		构成
		:param x:
		:return:
		'''
		if x.ndim == 1:
			n_features = 1
			x = x[:, np.newaxis]
		else:
			n_features = x.shape[1]
		# 构建分段数据
		self.XrangeMap = [[] for _ in range(n_features)]
		for idx in range(n_features):
			x_sorted = sorted(x[:, idx])
			for bin in range(1, self.max_bins):
				percent_val = np.percentile(x_sorted, (1.0 * bin / self.max_bins) * 100.0 // 1)
				self.XrangeMap[idx].append(percent_val)
			self.XrangeMap[idx]=sorted(list(set(self.XrangeMap[idx])))

