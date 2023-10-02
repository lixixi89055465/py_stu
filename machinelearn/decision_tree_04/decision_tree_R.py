# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/9/27 20:59
# @Author  : nanji
# @File    : decision_tree_C.py
# @Description :
import numpy as np
import sklearn.metrics as metrics
from machinelearn.decision_tree_04.utils.square_error_utils import SquareErrorUtils
from machinelearn.decision_tree_04.utils.tree_node_R import TreeNode_R
from machinelearn.decision_tree_04.utils.data_bin_wrapper import DataBinWrapper


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
                 min_samples_leaf=1, min_target_std=1e-3, min_impurity_decrease=0, max_bins=10):
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
        self.root_node: TreeNode_R() = None  # 分类决策树的根节点
        self.dbw = DataBinWrapper(max_bins=max_bins)  # 连续数据离散化对象
        self.dbw_XrangeMap = {}  # 存储训练样本连续特征分箱的段点

    def fit(self, x_train, y_train, sample_weight=None):
        '''
        回归决策树的创建，递归操作(分箱)
        @param x_train: 训练样本，ndarray,n*k
        @param y_train: 目标集 ：ndarray,(n,)
        @param sample_weight: 各样本的权重(n,)
        @return:
        '''
        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        n_sample, n_features = x_train.shape  # 训练样本的样本量和特征属性数目
        if sample_weight is None:
            sample_weight = np.asarray([1.0] * n_sample)
        self.root_node = TreeNode_R()  # 创建一个空树
        self.dbw.fit(x_train)
        x_train = self.dbw.transform(x_train)
        self._build_tree(1, self.root_node, x_train, y_train, sample_weight)
        # print(x_train)

    def _build_tree(self, cur_depth, cur_node: TreeNode_R, X, y, sample_weight):
        '''
        递归创建决策树算法,核心算法
        @param cur_depth: 递归划分后的树的深度
        @param cur_node: 递归后的当前根节点
        @param X: 递归划分后的训练样本
        @param y: 递归划分后的目标集合
        @param sample_weight: 递归划分后的各样本权重
        @return:
        '''
        n_samples, n_features = X.shape  # 当前样本子集中的样本量和特征属性数目
        # 计算当前节点的预测值，即加权平均值
        cur_node.y_hat = np.dot(sample_weight / np.sum(sample_weight), y)
        cur_node.n_samples = n_samples
        cur_node.square_error = ((y - np.mean(y)) ** 2).sum()
        # 判断停止的条件
        if np.sqrt(1.0 * cur_node.square_error / n_samples) <= self.min_target_std:
            # 如果为0，则表示当前样本集合为空，或特征属性集合为空，不足以划分
            return
        if n_samples < self.min_samples_split:  # 当前节点所包含的样本量不足以划分
            return
        if cur_depth > 100 or self.max_depth is not None and cur_depth > self.max_depth:
            return

            # 划分标准，选择最佳的划分特征及其取值
        # 寻找最佳的特征以及取值
        best_idx, best_idx_val, best_criterion_val = None, None, 0.0
        for idx in range(n_features):
            for idx_val in sorted(set(X[:, idx])):  # 每个特征的取值，分箱处理后数据
                region_x = (X[:, idx] <= idx_val).astype(int)  # 是当前取值f_val 就是1，否则就是0
                criterion_val = self.criterion_func(region_x, y, sample_weight)
                if criterion_val > best_criterion_val:
                    best_idx, best_idx_val = idx, idx_val  # 当前最佳的特征索引以及取值
                    best_criterion_val = criterion_val  # 最佳的划分标准值

        # 递归出口判断
        if best_idx is None:  # 当前属性为空，或者所有样本在所欲属性上取值相同，无法划分
            return
        if best_criterion_val <= self.min_impurity_decrease:  # 小于最小纯度阈值，不划分
            return
        # 满足划分条件，仅当前最佳特征索引和特征取值切分节点，填充树节点信息
        cur_node.criterion_val = best_criterion_val  # 最佳特征取值的标准
        cur_node.feature_idx = best_idx  # 最佳特征所在样本的索引
        cur_node.feature_val = best_idx_val  # 最佳特征取值
        # print('当前划分的特征索引：', best_idx, '\t取值：', best_index_val, '\t最佳标准值：', best_criterion_val)
        # print('当前节点的类别分布', target_dist)

        # 创建左子树，并递归创建以当前节点为子树根节点的左子树
        left_idx = np.where(X[:, best_idx] <= best_idx_val)  # 左子树所包含的样本子集索引
        if len(left_idx) >= self.min_sample_leaf:  # 小于叶子节点所包含的最小样本量，则标记为叶子节点
            left_child_node = TreeNode_R()  # 创建左子树空节点
            cur_node.left_child_node = left_child_node
            # 以当前节点为子树根节点，递归创建
            self._build_tree(cur_depth + 1, left_child_node, X[left_idx],
                             y[left_idx], sample_weight[left_idx])
        # 创建右子树，并递归创建以右子树为子树根节点的左子树
        right_idx = np.where(X[:, best_idx] > best_idx_val)  # 右子树所包含的样本子集索引
        if len(right_idx) >= self.min_sample_leaf:  # 小于叶子节点所包含的最少样本量，则标记为叶子节点
            right_child_Node = TreeNode_R()  # 创建右子树空树节点
            # 以当前节点为右子节点，递归创建
            cur_node.right_child_node = right_child_Node
            self._build_tree(cur_depth + 1, right_child_Node, X[right_idx],
                             y[right_idx], sample_weight[right_idx])

    def _search_tree_predict(self, cur_node: TreeNode_R, x_test):
        '''
        根据测试样本从根节点到叶子节点搜索路径，判定类别
        搜索：按照后续遍历
        @param cur_node: 单个测试样本
        @param x_test:
        @return:
        '''
        if cur_node.left_child_node and x_test[cur_node.feature_idx] <= cur_node.feature_val:
            return self._search_tree_predict(cur_node.left_child_node, x_test)
        elif cur_node.right_child_node and x_test[cur_node.feature_idx] > cur_node.feature_val:
            return self._search_tree_predict(cur_node.right_child_node, x_test)
        else:
            # 叶子节点：类别，包含有类别分布
            return cur_node.y_hat

    def predict_proba(self, x_test):
        '''
        预测样本x_test 的类别概率
        @param x_test:  测试样本ndarray,numpy数值运算
        @return:
        '''
        x_test = np.asarray(x_test)  # 避免传DataFrame,List
        if self.dbw_XrangeMap is None:
            raise ValueError('请选进行回归决策树的创建，然后预测...')
        x_test = self.dbw.transform(x_test)
        y_test_pred = []  # 用于存储测试样本呢的类别概率分布
        for i in range(x_test.shape[0]):
            y_test_pred.append(self._search_tree_predict(self.root_node, x_test[i]))
        return np.asarray(y_test_pred)

    def predict(self, x_test):
        '''
        预测测试样本的类别
        @param x_test: 测试样本
        @return:
        '''
        x_test = self.dbw.transform(x_test)
        y_pred = []
        for i in range(x_test.shape[0]):
            y_pred.append(self._search_tree_predict(self.root_node, x_test[i]))
        return np.asarray(y_pred)

    def cal_mse_r2(self, y_test, y_pred):
        '''
        模型预测的均方误差MSE 和判决系数r2
        :return:
        '''
        y_test, y_pred = np.asarray(y_test), np.asarray(y_pred)
        mse = ((y_test - y_pred) ** 2).mean()  # 均方误差
        r2 = 1 - ((y_pred - y_test) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
        return mse, r2

    def _prune_node(self, cur_node: TreeNode_R, alpha):
        '''
        递归剪枝，针对决策树中的内部节点，自底向上，逐个考察
        @param cur_node: 当前递归的决策树的内部节点
        @param alpha:
        @return:
        '''
        # 若左子树存在，递归左子树进行剪枝
        if cur_node.left_child_node:
            self._prune_node(cur_node.left_child_node, alpha)
        # 若右子树存在，递归右子树进行剪枝
        if cur_node.right_child_node:
            self._prune_node(cur_node.right_child_node, alpha)
        if cur_node.left_child_node is not None or cur_node.right_child_node is not None:
            for child_node in [cur_node.left_child_node, cur_node.right_child_node]:
                if child_node is None:
                    # 可能存在左右子树之一为空的情况，当左右子树划分的样本子集数小于min_samples_leaf
                    continue
                if child_node.left_child_node is not None or child_node.right_child_node is not None:
                    return
            # 计算剪枝前的损失值(平方误差)，2表示当前节点包含两个叶子节点
            pre_prune_value = 2 * alpha
            # for child_node in [cur_node.left_child_node, cur_node.right_child_node]:
            #     # 计算左右叶子节点的经验熵
            #     # 可能存在左右子树之一为空的情况，当左右子树划分的样本子集数小于min_samples_leaf
            #     if child_node is None:
            #         continue
            # for key, value in child_node.target_dist.items():  # 对每个叶子节点的类别分布
            #     pre_prune_value += -1 * child_node.n_samples * value *\
            #                        np.log(value) * child_node.weight_dist.get(key, 1.0)
            if cur_node and cur_node.left_child_node is not None:
                pre_prune_value += (
                    0.0 if cur_node.left_child_node.square_error is None else cur_node.left_child_node.square_error)
            if cur_node and cur_node.right_child_node is not None:
                pre_prune_value += (
                    0.0 if cur_node.right_child_node.square_error is None else cur_node.right_child_node.square_error)
            # 计算剪枝后的损失值，当前节点既是叶子节点，
            after_prune_value = alpha + cur_node.square_error
            # after_prune_value = alpha
            # for key, value in cur_node.target_dist.items():
            #     after_prune_value += -1 * cur_node.n_samples * value * np.log(value) * \
            #                          cur_node.weight_dist.get(key, 1.0)
            if after_prune_value <= pre_prune_value:  # 剪枝操作
                cur_node.left_child_node = None
                cur_node.right_child_node = None
                cur_node.feature_idx, cur_node.feature_val = None, None
                cur_node.square_error = None

    def prune(self, alpha=0.01):
        '''
        决策树后剪枝算法（李航）C(T)+alpha*|T|
        @param alpha:  剪纸阈值，权衡莫ing对训练数据的拟合成都与模型的复杂度
        @return:
        '''
        self._prune_node(self.root_node, alpha)
        return self.root_node
