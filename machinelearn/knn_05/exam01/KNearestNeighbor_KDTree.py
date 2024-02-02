# -*- coding: utf-8 -*-
# @Time : 2024/2/2 17:09
# @Author : nanji
# @Site : 
# @File : KNearestNeighbor_KDTree.py
# @Software: PyCharm 
# @Comment : 4. k近邻算法——自编算法实现（基于𝒌𝒅树）
import networkx as nx
import heapq
from collections import Counter
import numpy as np
from machinelearn.knn_05.exam01.KDTreeNode import KDTreeNode
from machinelearn.knn_05.exam01.DistanceUtils import DistanceUtils


class KNearestNeighbor_KDTree:
	def __init__(self, k=3, p=2, view_kdt=False):
		self.k = k
		self.p = p
		self.distance_utils = DistanceUtils(self.p)
		self.kdt_root = None
		self.k_dimension = 0
		self.k_neighbor = []
		self.view_kdt = view_kdt

	def fit(self, x_train, y_train):
		x_train, y_train = np.asarray(x_train), np.asarray(y_train)
		self.k_dimension = x_train.shape[1]
		sample_idx_array = np.arange(x_train.shape[0])
		self.kdt_root = self._build_kd_tree(x_train, y_train, sample_idx_array, 0)
		# if self.view_kdt:
		# 	self.draw_kd_tree()
