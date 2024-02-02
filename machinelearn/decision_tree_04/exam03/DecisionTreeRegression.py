from machinelearn.decision_tree_04.exam01.Entropy_Utils import Entropy_Utils
import numpy as np
from machinelearn.decision_tree_04.utils.data_bin_wrapper import DataBinWrapper
from machinelearn.decision_tree_04.exam02.TreeNode import TreeNode


class DecisionTreeRegression:
	'''
	CART算法创建回归树
	'''

	def __init__(self, criterion='mse', \
				 max_depth=None, \
				 min_samples_split=2, \
				 min_samples_leaf=1, \
				 min_std=1e-3, \
				 min_impurity_decrease=0, \
				 max_bins=10):
		'''
		:param criterion:
		:param max_depth:
		:param min_samples_split:
		:param min_samples_leaf:
		:param min_std:
		:param min_impurity_decrease:
		:param max_bins:
		'''
		self.criterion = criterion
		if criterion == "mse":
			self.criterion_func = Entropy_Utils().square_error_gain
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.min_samples_leaf = min_samples_leaf
		self.min_std = min_std
		self.min_impurity_decrease = min_impurity_decrease
		self.root_node: TreeNode() = None  # 树结点
		self.dbw = DataBinWrapper(max_bins=max_bins)  # 连续数据的分箱处理

	@staticmethod
	def cal_mse_r2(y_test, y_pred):
		'''

		:param y_test:
		:param y_pred:
		:return:
		'''
		y_test, y_pred = np.asarray(y_test), np.asarray(y_pred)
		y_pred, y_test = y_pred.reshape(-1), y_test.reshape(-1)
		mse = ((y_pred - y_test) ** 2).mean()  # 均方误差
		r2 = 1 - ((y_test - y_pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
		return mse, r2

	def fit(self, x_train, y_train, sample_weight=None):
		'''

		:param x_train:
		:param y_train:
		:param sample_weight:
		:return:
		'''
		n_samples = x_train.shape[0]
		if sample_weight is None:
			sample_weight = np.asarray([1.0] * n_samples)
		if len(sample_weight) != n_samples:
			raise Exception('sample weight size error', len(sample_weight))
		self.root_node = TreeNode()
		self.dbw.fit(x_train)
		x_train_ = self.dbw.transform(x_train)
		# 递归构建树
		self._build_tree(1, self.root_node, x_train, y_train, sample_weight)

	def prune(self, alpha=.01):
		'''
		决策树剪枝C(T)+alpha*|T|
		:param alpha:
		:return:
		'''
		self._prune_node(self.root_node, alpha)

	def _build_tree(self, cur_depth, cur_node: TreeNode, X, y, sample_weight):
		'''
		递归进行特征选择，构建树
		:param cur_depth:
		:param cur_node:
		:param X:
		:param y:
		:param sample_weight:
		:return:
		'''
		n_samples, n_features = X.shape  # d当前训练样本呢量和特征数目
		cur_node.y_hat = np.dot(sample_weight / np.sum(sample_weight), y)
		cur_node.num_samples = n_samples  # 当前节点所包含的样本量
		cur_node.square_error = ((y - np.mean(y)) ** 2).sum()  # 误差平方和
		if np.sqrt(cur_node.square_error / n_samples) <= self.min_std:  # 最小MSE
			return
		if n_samples < self.min_samples_split:  # 最小划分点
			return
		if self.max_depth is not None and cur_depth > self.max_depth:
			return
		best_idx, best_idx_val, best_criterion_val = None, None, 0
		for idx in range(n_features):
			for idx_val in sorted(set(X[:, idx])):
				region_x = (X[:, idx] <= idx_val).astype(int)
				criterion_val = self.criterion_func(region_x, y, sample_weight)
				if best_criterion_val < criterion_val:
					best_criterion_val = criterion_val
					best_idx_val = idx_val
					best_idx = idx
		if best_idx is None:
			return
		if best_criterion_val <= self.min_impurity_decrease:
			return
		cur_node.feature_idx, cur_node.feature_val = best_idx, best_idx_val
		selected_x = X[:, best_idx]  # 当前选择的最佳特征样本
		# 递归创建左孩子节点
		left_idx = np.where(selected_x <= best_idx_val)
		# 如果切分后的点太少，以至于不能左叶子节点，则停止分割
		if len(left_idx[0]) >= self.min_samples_leaf:
			left_child_node = TreeNode()
			cur_node.left_child_node = left_child_node
			self._build_tree(cur_depth + 1, \
							 left_child_node, \
							 X[left_idx], \
							 y[left_idx], \
							 sample_weight[left_idx])
		right_idx = np.where(selected_x > best_idx_val)
		if len(right_idx[0]) >= self.min_samples_leaf:
			right_child_node = TreeNode()
			cur_node.right_child_node = right_child_node
			self._build_tree(cur_depth + 1, \
							 right_child_node, \
							 X[right_idx],\
							 y[right_idx], \
							 sample_weight[right_idx])

	def predict(self, x_test):
		'''
		计算结果概率分布
		:param x_test:
		:return:
		'''
		x_test = np.asarray(x_test)
		x_test = self.dbw.transform(x_test)
		y_pred = []
		for i in range(x_test.shape[0]):
			y_pred.append(self._search_node(self.root_node, x_test[i]))
		return np.asarray(y_pred)

	def _prune_node(self, cur_node: TreeNode, alpha):
		'''
		回归树递归剪枝，按照后续排序
		:param root_node:
		:param alpha:
		:return:
		'''
		if cur_node.left_child_node:
			self._prune_node(cur_node.left_child_node, alpha)
		if cur_node.right_child_node:
			self._prune_node(cur_node.right_child_node, alpha)
		# 对当前节点剪枝
		if cur_node.left_child_node or cur_node.right_child_node:
			for child_node in [cur_node.left_child_node, cur_node.right_child_node]:
				if child_node is None:
					continue
				if child_node.left_child_node or child_node.right_child_node:
					return
			pre_prune_value = alpha * 2
			if child_node and child_node.left_child_node is not None:
				pre_prune_value += (.0 if cur_node.left_child_node.square_error is None \
										else cur_node.left_child_node.square_error)
			if child_node and child_node.right_child_node is not None:
				pre_prune_value += (.0 if cur_node.right_child_node.square_error is None \
										else cur_node.right_child_node.square_error)
			after_prune_value = alpha + cur_node.square_error
			if after_prune_value <= pre_prune_value:
				cur_node.left_child_node, cur_node.right_child_node = None, None
				cur_node.feature_idx, cur_node.feature_val = None, None
				cur_node.square_error = None

	def _search_node(self, cur_node: TreeNode, x):
		'''
		:param cur_node:
		:param x:
		:return:
		'''
		if cur_node.left_child_node and x[cur_node.feature_idx] <= cur_node.feature_val:
			return self._search_node(cur_node.left_child_node, x)
		elif cur_node.right_child_node and x[cur_node.feature_idx] > cur_node.feature_val:
			return self._search_node(cur_node.right_child_node, x)
		else:
			return cur_node.y_hat


import matplotlib.pyplot as plt

obj_fun = lambda x: np.sin(x)  #
np.random.seed(0)
n = 100
x = np.linspace(0, 10, n)
target = obj_fun(x) + 0.3 * np.random.random(size=n)
data = x.reshape((-1, 1))
tree = DecisionTreeRegression(max_bins=50, max_depth=10)
tree.fit(data, target)
x_test = np.linspace(0, 10, 200)
y_test = obj_fun(x_test)
y_pred = tree.predict(x_test.reshape(-1, 1))
mse, r2 = tree.cal_mse_r2(y_test, y_pred)
plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.scatter(data, target, s=15, c='k', label='Raw Data')
plt.plot(x_test, y_pred, color='r', lw=1.5, label='Fit model')
plt.xlabel('$x$', fontdict={'fontsize': 12})
plt.ylabel('$y(x)=2e^{-x}sin(x)$', fontdict={'fontsize': 12})
plt.legend(frameon=False)
plt.grid(ls=':')
plt.title('Decision Tree Regression Model(UnPrune) of '
		  'Test Samples \n MSE= %.5e, R2=%.5f' % (mse, r2))
tree.prune(0.2)
plt.subplot(122)
y_pred = tree.predict(x_test.reshape(-1, 1))
mse, r2 = tree.cal_mse_r2(y_test, y_pred)
plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.scatter(data, target, s=15, c='k', label='Raw Data')
plt.plot(x_test, y_pred, color='r', lw=1.5, label='Fit Model')
plt.xlabel('$x$', fontdict={'fontsize': 12})
plt.ylabel('$y$', fontdict={'fontsize': 12})
plt.legend(frameon=False)
plt.grid(ls=':')
plt.title(f'Decision Tree Regression Model(UnPrune) of '
		  f'Tes sample MSE={mse},R2={r2}')
tree.prune(.2)  # 剪枝
plt.subplot(122)
y_pred = tree.predict(x_test.reshape(-1, 1))
mse, r2 = tree.cal_mse_r2(y_test, y_pred)
plt.savefig('DecisionTreeRegression.py.png')
plt.show()
