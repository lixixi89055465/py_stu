# -*- coding: utf-8 -*-
# @Time : 2024/2/2 17:09
# @Author : nanji
# @Site : 
# @File : KNearestNeighbor_KDTree.py
# @Software: PyCharm 
# @Comment : 4. k近邻算法——自编算法实现（基于𝒌𝒅树）
import numpy as np
import heapq
from machinelearn.knn_05.exam01.DistanceUtils import DistanceUtils
from machinelearn.knn_05.exam01.KDTreeNode import KDTreeNode


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
	def _build_kd_tree(self, x_train, y_train, idx_array, depth):
		'''
		递归构造kd树：kd树是二叉树，表示对k维空间的一个划分。
		构造kd树相当于不断地用垂直于坐标轴的超平面将k维空间切分，
		构成一系列的k维超矩形区域。
		kd树的每个节点对应于一个k维超矩形区域。
		:param x_train:训练集
		:param y_train:目标值
		:param idx_array:
		:param depth:
		:return:
		'''
		if x_train.shape[0] == 0:
			return None
		split_dimension = depth % self.k_dimension
		sorted(x_train, key=lambda x: x[split_dimension])
		median_idx = x_train.shape[0] // 2
		median_instance = x_train[median_idx]
		left_instances, right_instances = x_train[:median_idx], x_train[median_idx + 1:]
		left_labels, right_labels = y_train[:median_idx], y_train[median_idx + 1:]
		left_n, right_n = idx_array[:median_idx], idx_array[median_idx + 1:]

		left_child = self._build_kd_tree(left_instances, left_labels, left_n, depth + 1)
		right_child = self._build_kd_tree(right_instances, right_labels, right_n, depth + 1)
		kdt_new_node = KDTreeNode(median_instance, \
								  y_train[median_idx], \
								  idx_array[median_idx],
								  split_dimension,
								  left_child, right_child, depth)

	def search_kd_tree(self, kd_tree, x_test):
		if kd_tree is None:
			return
		else:
			distance = self.distance_utils.distance_func(kd_tree.instancec_node, x_test)
			self.k_neighbor.reverse()
			if (len(self.k_neighbor) < self.k) or (distance < self.k_neighbor[0]['distance']):
				self.search_kd_tree(kd_tree.left_child, x_test)
				self.sarch_kd_tree(kd_tree.right_child, x_test)
				self.k_neighbor.append({
					'node': kd_tree.instancec_node,
					'label': kd_tree.instance_label,
					'distance': distance
				})
				self.k_neighbor = heapq.nsmallest(self.k, self.k_neighbor, \
												  key=lambda d: d['distance'])

	def predict(self,x_test):
		'''
		对测试样本进行类别预测
		:param x_test: 测试样本
		:return:
		'''
		x_test=np.asarray(x_test)
