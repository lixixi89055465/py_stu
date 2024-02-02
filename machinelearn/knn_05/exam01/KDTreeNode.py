# -*- coding: utf-8 -*-
# @Time : 2024/2/2 16:18
# @Author : nanji
# @Site : 
# @File : KDTreeNode.py
# @Software: PyCharm 
# @Comment : 
import os
import numpy as np


class KDTreeNode:
	def __init__(self, instance_node=None, \
				 instance_label=None, \
				 instance_idx=None, \
				 split_feature=None, \
				 left_child=None, \
				 right_child=None, \
				 kdt_depth=None):
		'''

		:param instance_node:
		:param instance_label:
		:param split_feature:
		:param left_child:
		:param right_child:
		:param kdt_depth:
		'''
		self.instance_node = instance_node
		self.instance_label = instance_label
		self.instance_idx = instance_idx
		self.split_feature = split_feature
		self.left_child = left_child
		self.right_child = right_child
		self.kdt_depth = kdt_depth

