import numpy as np


class TreeNode(object):
	def __init__(self, feature_idx: int = None, \
				 feature_val=None, y_hat=None, \
				 square_error=None, left_child_node=None,\
				 right_child_node=None, \
				 num_samples: int = None):
		self.feature_idx = feature_idx  # 特征id
		self.feature_val = feature_val  # 特征取值
		self.y_hat = y_hat  # 预测值
		self.square_error = square_error  # 当前结点的平方误差
		self.left_child_node = left_child_node  # 左孩子h点
		self.right_child_node = right_child_node  # 右孩子结点
		self.num_samples = num_samples

