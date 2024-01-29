# -*- coding: utf-8 -*-
# @Time : 2024/1/28 16:26
# @Author : nanji
# @Site : 
# @File : tree_node_R.py
# @Software: PyCharm 
# @Comment : https://www.bilibili.com/video/BV1Nb4y1s7nV/?p=46&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=50305204d8a1be81f31d861b12d4d5cf

class TreeNode_R:
	'''
	决策树回归算法，树的节点信息封装，实体类:setXXX(),getXXX()
	'''

	def __init__(self, feature_idx, feature_val, y_hat=None, square_error=None,\
				 n_samples=None, left_child_node=None, right_child_Node=None):
		'''
		决策树节点信息封装
		:param feature_idx:特征索引，如果指定属性名称，可以按照索引取值
		:param feature_val:特征取值
		:param n_samples:当前节点所包含的样本量
		:param weight_dist: 当前节点的预测值
		:param left_child_node: 左子树
		:param right_child_Node:右子树
		'''
		self.feature_idx = feature_idx
		self.feature_val = feature_val
		self.square_error = square_error
		self.y_hat = y_hat
		self.left_child_node = left_child_node
		self.right_child_node = right_child_Node

	def level_order(self):
		'''
		按层次遍历树
		:return:
		'''
