# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/9/27 20:59
# @Author  : nanji
# @File    : decision_tree_C.py
# @Description :
import time

import numpy as np

from machinelearn.decision_tree_04.utils.data_bin_wrapper import DataBinWrapper
from machinelearn.decision_tree_04.utils.entropy_utils import EntropyUtils
from machinelearn.decision_tree_04.utils.tree_node import TreeNode_C


class DecisionTreeClassifier:
	'''
	分类决策树算法实现: 无论是ID3,C4.5或CART,统一按照二叉树构造
	1.划分标准：信息增益（率),基尼指数增益，都按照最大值选择特征属性
	2.创建决策树fit(),递归算法实现，注意出口条件
	3.预测predict_proba(),predict(),-->对树的搜索，从根到叶
	4.数据的预处理操作，尤其是连续数据的离散化，分箱
	5.剪枝处理
	'''

	def __init__(self, criterion='cart', is_feature_all_R=False, class_num=None,
				 dbw_feature_idx=None, max_depth=None, min_samples_split=2,
				 min_samples_leaf=1, min_impurity_decrease=0, max_bins=10, dbw_XrangeMap=None):
		self.utils = EntropyUtils()  # 结点划分类
		self.criterion = criterion  # 结点的划分标准
		if criterion.lower() == 'cart':
			self.criterion_func = self.utils.gini_gain  # 基尼指数增益
		elif criterion.lower() == 'c45':
			self.criterion_func = self.utils.info_gain_rate  # 信息增益率
		elif criterion.lower() == 'id3':
			self.criterion_func = self.utils.info_gain  # 信息增益
		else:
			raise ValueError("参数criterion仅限cart、c45或id3...")
		self.is_feature_all_R = is_feature_all_R  # 所有样本呢特征是否全是连续数据
		self.dbw_feature_idx = dbw_feature_idx  # 混合类型数据，可指定连续特征属性的索引
		self.max_depth = max_depth  # 树的最大深度，不传参，则一直划分下去
		self.min_sample_split = min_samples_split  # 最小的划分节点的样本量，小于则不划分
		self.min_sample_leaf = min_samples_leaf  # 叶子节点所包含的最小样本量，剩余的样本小于这个值，标记叶子节点
		self.min_impurity_decrease = min_impurity_decrease  # 最小结点不纯度减少值，小于这个值，不足以划分
		self.max_bins = max_bins  # 连续数据的分箱数，越大，则划分越细
		self.root_node: TreeNode_C() = None  # 分类决策树的根节点
		self.dbw = DataBinWrapper(max_bins=max_bins)  # 连续数据离散化对象
		self.dbw_XrangeMap = {}  # 存储训练样本连续特征分箱的段点
		self.class_num = class_num
		self.n_features = None
		self.prune_nums = 0
		if class_num is not None:
			self.class_values = np.arange(class_num)  # 样本的类别取值

	def _data_bin_wrapper(self, x_samples):
		'''
		针对特征的连续的特征属性索引dbw_feature_idx,分别进行分箱,
		考虑测试样本与训练样本使用同一个XrangeMap
		@param X_samples: 样本：即可是训练样本,也可以是测试样本
		@return:
		'''
		self.dbw_feature_idx = np.asarray(self.dbw_feature_idx)
		x_samples_prob = []
		if not self.dbw_XrangeMap:
			for i in range(x_samples.shape[1]):
				if i in self.dbw_feature_idx:
					self.dbw.fit(x_samples[:, i])
					self.dbw_XrangeMap[i] = self.dbw.XrangeMap
					x_samples_prob.append(self.dbw.transform(x_samples[:, i]))
				else:
					x_samples.append(x_samples[:, i])
		else:
			for i in range(x_samples.shape[1]):
				if i in self.dbw_feature_idx:
					x_samples_prob.append(self.dbw.transform(x_samples[:, i], \
															 self.dbw_XrangeMap[i]))
				else:
					x_samples_prob.append(x_samples[:, i])
		return np.asarray(x_samples_prob).T

	def fit(self, x_train, y_train, sample_weight=None):
		'''
		决策树的创建，递归操作
		@param x_train: 训练样本，ndarray,n*k
		@param y_train: 目标集 ：ndarray,(n,)
		@param sample_weight: 各样本的权重(n,)
		@return:
		'''
		x_train, y_train = np.asarray(x_train), np.asarray(y_train)
		if self.class_num is None:
			self.class_values = np.unique(y_train)  # 样本的类别取值
		n_sample, self.n_features = x_train.shape  # 训练样本的样本量和特征属性数目
		if sample_weight is None:
			sample_weight = np.asarray([1.0] * n_sample)

		self.root_node = TreeNode_C()
		if self.is_feature_all_R:  # 全部是连续数据
			self.dbw.fit(x_train)
			x_train = self.dbw.transform(x_train)
		elif self.dbw_feature_idx:
			x_train = self._data_bin_wrapper(x_train)
		# 递归构建树
		time_start = time.time()
		self._build_tree(1, self.root_node, x_train, y_train, sample_weight)
		time_end = time.time()
		print('决策树模型递归构建完成，耗时 : %f second' % (time_end - time_start))

	def _build_tree(self, cur_depth, cur_node: TreeNode_C, x_train, y_train, sample_weight):
		'''
		递归创建决策树算法,核心算法
		@param cur_depth: 递归划分后的树的深度
		@param cur_node: 递归后的当前根节点
		@param x_train: 递归划分后的训练样本
		@param y_train: 递归划分后的目标集合
		@param sample_weight: 递归划分后的各样本权重R
		@return:
		'''
		n_samples, n_features = x_train.shape
		target_dist, weight_dist = {}, {}
		class_labels = np.unique(y_train)
		for label in class_labels:
			target_dist[label] = len(y_train[y_train == label]) / n_samples
			weight_dist[label] = np.mean(sample_weight[y_train == label])
		cur_node.target_dist = target_dist
		cur_node.weight_dist = weight_dist
		cur_node.n_samples = n_samples
		if len(target_dist) <= 1:
			return
		if n_samples < self.min_sample_split:
			return
		if self.max_depth is not None and cur_depth > self.max_depth:
			return
		best_idx, best_index_val, best_criterion_val = None, None, 0
		for k in range(n_features):
			for f_val in set(x_train[:, k]):
				feat_k_values = (x_train[:, k] == f_val).astype(int)
				criterion_val = self.criterion_func(feat_k_values, y_train, sample_weight)
				if criterion_val > best_criterion_val:
					best_criterion_val = criterion_val
					best_idx, best_index_val = k, f_val
		if best_idx is None:
			return
		if best_criterion_val <= self.min_impurity_decrease:
			return
		cur_node.feature_idx = best_idx
		cur_node.feature_val = best_index_val
		cur_node.criterion_val = best_criterion_val
		selected_x = x_train[:, best_idx]
		left_index = np.where(selected_x == best_index_val)
		if len(left_index[0]) >= self.min_sample_leaf:
			left_child_node = TreeNode_C()
			cur_node.left_child_node = left_child_node
			self._build_tree(cur_depth + 1, left_child_node, \
							 x_train[left_index], \
							 y_train[left_index], \
							 sample_weight[left_index])
		right_index = np.where(selected_x != best_index_val)
		if len(right_index[0]) >= self.min_sample_leaf:
			right_child_node = TreeNode_C()
			cur_node.right_child_node = right_child_node
			self._build_tree(
				cur_depth + 1, \
				right_child_node, \
				x_train[right_index], \
				y_train[right_index], \
				sample_weight[right_index])

	def _search_node(self, cur_node: TreeNode_C, x_test, class_num):
		'''
		:param cur_node:
		:param x_test:
		:param class_num:
		:return
		'''
		if cur_node.left_child_node and x_test[cur_node.feature_idx] == cur_node.feature_val:
			return self._search_node(cur_node.left_child_node, x_test, class_num)
		elif cur_node.right_child_node and x_test[cur_node.feature_idx] != cur_node.feature_val:
			return self._search_node(cur_node.right_child_node, x_test, class_num)
		else:
			class_p = np.zeros(class_num)
			for c in range(class_num):
				class_p[c] = cur_node.target_dist.get(c, 0) * cur_node.weight_dist.get(c, 1.0)
			class_p = class_p / np.sum(class_p)
			return class_p

	def predict_probability(self, x_test, root_node=None):
		if self.is_feature_all_R:
			x_test = self.dbw.transform(x_test)
		elif self.dbw_feature_idx is not None:
			x_test = self._data_bin_wrapper(x_test)
		time_start = time.time()
		prob_dist = []
		class_num = len(self.root_node.target_dist)
		for i in range(x_test.shape[0]):
			prob_dist.append(self._search_node(self.root_node, x_test[i], class_num))
		time_end = time.time()
		return np.asarray(prob_dist)

	def predict(self, x_test):
		return np.argmax(self.predict_probability(x_test), axis=1)


from sklearn.datasets import load_breast_cancer


def target_encoding(y):
	target_dict = {}
	y_unique = set(y)
	for i, k in enumerate(y_unique):
		target_dict[k] = i
	n_samples, n_class = len(y), len(set(y))
	target = -1.0 / (n_class - 1) * np.ones((n_samples, n_class))
	for i in range(n_samples):
		target[i, target_dict[y[i]]] = 1
	return target, target_dict


from sklearn.model_selection import train_test_split

bc_data = load_breast_cancer()
feature_names = bc_data.feature_names
X, y = bc_data.data, bc_data.target
# y, y_labels_dict = target_encoding(y)
X_train, X_test, y_train, y_test = \
	train_test_split(X, y, \
					 test_size=0.3, \
					 random_state=22, \
					 stratify=y)
tree = DecisionTreeClassifier(is_feature_all_R=True, \
							  max_bins=10, \
							  criterion='c45', \
							  max_depth=10)
tree.fit(X_train, y_train)
# tree.out_decision_tree(feature_names=feature_names)
y_test_pred = tree.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_test_pred))
