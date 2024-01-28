# -*- coding: utf-8 -*-
# @Time : 2024/1/28 16:55
# @Author : nanji
# @Site : 
# @File : decision_tree_R.py
# @Software: PyCharm 
# @Comment : https://www.bilibili.com/video/BV1Nb4y1s7nV/?p=47&spm_id_from=pageDriver&vd_source=50305204d8a1be81f31d861b12d4d5cf

import numpy as np
from machinelearn.decision_tree_04.exam02.SquareErrorUtils import SquareErrorUtils
from machinelearn.decision_tree_04.exam02.tree_node_R import TreeNode_R
from machinelearn.decision_tree_04.data_bin_wrapper import DataBinWrapper


class DecisionTreeRegression:
	'''
	   回归决策树CART算法实现: 按照二叉决策树构造
	   1.划分标准：平方误差最小化
	   2.创建决策树fit(),递归算法实现，注意出口条件
	   3.预测predict_proba(),predict(),-->对树的搜索，从根到叶
	   4.数据的预处理操作，尤其是连续数据的离散化，分箱
	   5.剪枝处理
	   '''

	def __init__(self, criterion='mse', max_depth=None, min_samples_split=2,
				 min_samples_leaf=1, min_target_std=1e-3, \
				 min_impurity_decrease=0, max_bins=10):
		self.utils = SquareErrorUtils()  # 结点划分类
		self.criterion = criterion  # 结点的划分标准
		if criterion.lower() == 'mse':
			self.criterion_func = self.utils.square_error_gain  # 平方误差增益
		else:
			raise ValueError("参数criterion仅限mse...")
		self.min_target_std = min_target_std  # 最小的样本标准值方差，小于阈值不划分
		self.max_depth = max_depth  # 树的最大深度，不传参，则一直划分下去
		self.min_samples_split = min_samples_split  # 最小的划分节点的样本量，小于则不划分
		self.min_sample_leaf = min_samples_leaf  # 叶子节点所包含的最小样本量，剩余的样本小于这个值，标记叶子节点
		self.min_impurity_decrease = min_impurity_decrease  # 最小结点不纯度减少值，小于这个值，不足以划分
		self.max_bins = max_bins  # 连续数据的分箱数，越大，则划分越细
		self.root_node: TreeNode_R() = None  # 回归决策树的根节点
		self.dbw = DataBinWrapper(max_bins=max_bins)  # 连续数据离散化对象
		self.dbw_XrangeMap = {}  # 存储训练样本连续特征分箱的段点

	def fit(self, x_train, y_train, sample_weight=None):
		'''
		回归决策树的创建，递归操作前的必要信息处理（分箱）
		:param x_train: 训练样本:ndarray,n*k
		:param y_train: 目标集:ndarray,(n,)
		:param sample_weight:  各样本的权重(n,)
		:return:
		'''
		x_train, y_train = np.asarray(x_train), np.asarray(y_train)
		self.class_values = np.unique(y_train)  # 样本的类别取值
		n_samples, n_features = x_train.shape  # 训练样本的样本量和特征属性数目
		if sample_weight is None:
			sample_weight = np.asarray([1.0] * n_samples)
		self.root_node = TreeNode_R()  # 创建一个空树
		self.dbw.fit(x_train)
		x_train = self.dbw.transform(x_train)
		self._build_tree(1, self.root_node, x_train, y_train, sample_weight)

	def _build_tree(self, cur_depth, cur_node, x_train, y_train, sample_weight):
		'''
		递归创建回归决策树算法，核心部分。按先序（中序、后序）创建的
		:param cur_depth: 递归划分后的树的深度
		:param cur_node: 递归后的当前节点
		:param x_train: 递归划分后的训练样本
		:param y_train: 递归划分后的目标集合
		:param sample_weight: 递归划分后的各样本权重
		:return:
		'''
		n_samples, n_features = x_train.shape  # 当前样本了集中的样本量和特征属性数目
		# 计算当前数节点的预测值，即加权平均值
		cur_node.y_hat = np.dot(sample_weight / np.sum(sample_weight), y_train)
		cur_node.n_samples = n_samples
		# 递归出口的判断
		cur_node.square_error = ((y_train - y_train.mean()) ** 2).sum()
		# 所有的样本目标值较为集中，样本方差非常小,不足以划分
		if cur_node.square_error <= self.min_target_std:
			return
		if n_samples < self.min_samples_split:
			return
		if self.max_depth is not None and cur_depth > self.max_depth:
			return
		# 划分标准，选择最佳的划分特征及其取值
		best_idx, best_val, best_criterion_val = None, None, 0.0
		for k in range(n_features):
			for f_val in np.unique(x_train[:, k]):
				region_x = (x_train[:, k] < f_val).astype(int)
				criterion_val = self.criterion_func(region_x, y_train, sample_weight)
				if criterion_val > best_criterion_val:
					best_criterion_val = criterion_val
					best_idx, best_val = k, f_val
		if best_idx is None:
			return
		if best_criterion_val <= self.min_impurity_decrease:
			return
		cur_node.criterion_val = best_criterion_val
		cur_node.feature_idx = best_idx
		cur_node.feature_val = best_val
		# 创建左子树，并递归创建以挡墙节点为子树根节点的左子树
		left_idx = np.where(x_train[:, best_idx] <= best_val)
		if len(left_idx) >= self.min_samples_leaf:
			left_child_node = TreeNode_R()  # 创建左子树节点
			cur_node.left_child_node = left_child_node
			self._build_tree(cur_depth + 1, \
							 left_child_node, \
							 x_train[left_idx], \
							 y_train[left_idx], \
							 sample_weight[left_idx])
		# 闯将右子树,并递归创建以当前节点为子树跟节点的左子树
		right_idx = np.where(x_train[:, best_idx] != best_val)
		if len(right_idx) > self.min_samples_leaf:
			right_child_node = TreeNode_R()
			cur_node.right_child_node = right_child_node
			self._build_tree(cur_depth + 1, right_child_node, \
							 x_train[right_idx], \
							 y_train[right_idx], \
							 sample_weight[right_idx])

	def predict_proba(self, x_test):
		'''
		预测测试样本x_test的类别概率。
		:param x_test: 测试样本ndarray,numpy数值运算
		:return:
		'''
		x_test = np.asarray(x_test)
		if self.dbw.XrangeMap is None:
			raise ValueError('请先进行决策树的创建，然后预测.......')
		x_test = self._data_bin_wrapper(x_test)
		prob_dist = []
		for i in range(x_test.shape[0])
			prob_dist.append(self._search_tree_predict(self.root_node, x_test[i]))
		return np.asarray(prob_dist)
