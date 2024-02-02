# -*- coding: utf-8 -*-
# @Time : 2024/1/28 16:26
# @Author : nanji
# @Site : 
# @File : TreeNode.py
# @Software: PyCharm 
# @Comment : 回归决策树——算法实现

class TreeNode_R:
	'''
	树节点，用于存储节点信息以及关联子节点
	'''

	def __init__(self, feature_idx, \
				 feature_val, \
				 y_hat=None, \
				 square_error=None, \
				 left_child_node=None, \
				 right_child_Node=None, \
				 num_samples=0):
		'''
		决策树节点信息封装
		:param feature_idx:特征索引，如果指定属性名称，可以按照索引取值
		:param feature_val:特征取值
		:param y_hat:预测值
		:param square_error:当前节点所包含的样本量
		:param weight_dist: 当前节点的预测值
		:param left_child_node: 左子树
		:param right_child_Node:右子树
		:param num_samples: 样本量
		'''
		self.feature_idx = feature_idx
		self.feature_val = feature_val
		self.square_error = square_error
		self.y_hat = y_hat
		self.left_child_node = left_child_node
		self.right_child_node = right_child_Node
		self.num_samples = num_samples
