# -*- coding: utf-8 -*-
# @Time : 2024/2/2 17:08
# @Author : nanji
# @Site : 
# @File : DistanceUtils.py
# @Software: PyCharm 
# @Comment : 

import numpy as np

class DistanceUtils:
	def __init__(self, p=2):
		self.p = p

	def distance_func(self, xi, xj):
		if self.p == 1 or self.p == 2:
			return (((xi - xj) ** self.p).sum())
		elif self.p is np.inf:
			return np.max(np.abs(xi - xj))
		else:
			raise ValueError('目前仅支持p=1,p=2,p=np.inf 三种距离!!')

