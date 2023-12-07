from machinelearn.decision_tree_04.exam01.Entropy_Utils import Entropy_Utils
import numpy as np
from machinelearn.decision_tree_04.utils.data_bin_wrapper import DataBinWrapper
from machinelearn.decision_tree_04.exam02.TreeNode import TreeNode


class DecisionTreeRegression:
	'''
	CART算法创建回归树
	'''

	def __init__(self, criterion='mse', max_depth=None, \
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
