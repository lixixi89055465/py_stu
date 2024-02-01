# -*- coding: utf-8 -*-
# @Time    : 2023/9/24 下午11:29
# @Author  : nanji
# @Site    : 
# @File    : data_bin_wrapper.py
# @Software: PyCharm 
# @Comment :

import numpy as np


class DataBinWrapper:
	'''
	连续特征数据的离散化，分箱(分段)操作,根据用户传参_bins,计算分位数，以分位数（分箱）分段
	然后根据样本特征取值所在的区间段（哪个箱）位置索引标记当前值
	1.fit(x)根据样本进行分箱
	2.transform(x)根据已存在的箱，把数据分成max_bins类
	'''

	def __init__(self, max_bins=10):
		self.max_bins = max_bins  # 分箱数
		self.XrangeMap = None  # 箱（区间数）

	def fit(self, x):
		if x.ndim==1:
			n_features=1
			x=x[:,np.newaxis]
		else:
			n_features=x.shape[1]
		self.XrangeMap=[[] for _ in range(n_features)]
		for idx in range(n_features):
			x_sorted=sorted(x[:,idx])
			for bin in range(1,self.max_bins):
				percent_val=np.percentile(x_sorted,(1.0*bin/self.max_bins*100.0//1))
				self.XrangeMap[idx].append(percent_val)
			self.XrangeMap[idx]=sorted(list(set(self.XrangeMap[idx])))


	def transform(self, x_samples, XrangeMap=None):
		'''
		根据已存在的箱，把数据分成max_bins类
		:param x_samples: 样本(二维数组n*k),或一个特征属性的数据（二维数组n*1)
		:return:
		'''
		if x_samples.ndim == 1:
			return np.asarray([np.digitize(x_samples, self.XrangeMap[0])]).reshape(-1)
		else:
			return np.asarray([np.digitize(x_samples[:, i], self.XrangeMap[i]) for i in range(x_samples.shape[1])]).T



if __name__ == '__main__':
	# x = np.random.randint(10, 80, 20)
	x = np.random.randn(10, 5)
	dbw = DataBinWrapper(max_bins=5)
	print(x)
	dbw.fit(x)
	print(dbw.XrangeMap)
	print('1' * 100)
	print(dbw.transform(x))
