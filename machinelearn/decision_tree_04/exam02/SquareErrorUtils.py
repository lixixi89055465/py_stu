# -*- coding: utf-8 -*-
# @Time : 2024/1/28 15:24
# @Author : nanji
# @Site : 
# @File : SquareErrorUtils.py
# @Software: PyCharm 
# @Comment :

import numpy as np


class SquareErrorUtils:
	'''
	平方误差最小化准则选择其中最优的一个作为切分点
	'''

	@staticmethod
	def square_error(y, sample_weight):
		'''
		平方误差
		:param y:
		:param sample_weight:
		:return:
		'''
		y = np.asarray(y)
		return np.sum((y - y.mean()) ** 2 * sample_weight)

	def cond_square_error(self, x, y, sample_weight):
		'''
		计算根据某个特征x划分的趋于中y的误差值
		:param x: 某个特征划分区域所包含的样本
		:param y: x对应的目标值
		:param sample_weight:  当前 x的权重
		:return:
		'''
		x, y = np.asarray(x), np.asarray(y)
		error = 0.
		for x_val in set(x):
			x_idx = np.where(x == x_val)
			new_y = y[x_idx]
			new_sample_weight = sample_weight[x_idx]
			error += self.square_error(new_y, new_sample_weight)
		return error

	@staticmethod
	def _set_sample_weight(sample_weight, n_sample):
		'''
		扩展到集成学习，此处为样本权重的设置

		:param sample_weight:各样本的权重
		:param n_sample: 样本量
		:return:
		'''
		if sample_weight is None:
			sample_weight = np.asarray([1.0] * n_sample)
		return sample_weight

	def square_error_gai(self,x,y,sample_weight=None):
		'''
		平方误差
		:param x:
		:param y:
		:param sample:
		:return:
		'''
		sample_weight=self._set_sample_weight(sample_weight,len(x))
		return self.square_error(y,sample_weight)-self.cond_square_error(x,y,sample_weight)

