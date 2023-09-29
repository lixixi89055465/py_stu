# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/9/27 20:59
# @Author  : nanji
# @File    : decision_tree_C.py
# @Description :
import numpy as np
from machinelearn.decision_tree_04.utils.entropy_utils import EntropyUtils
from machinelearn.decision_tree_04.utils.tree_node import TreeNode_C
from machinelearn.decision_tree_04.utils.data_bin_wrapper import DataBinWrapper


class DecisionTreeClassifier:
    '''
    分类决策树算法实现: 无论是ID3,C4.5或CART,统一按照二叉树构造
    1.划分标准：信息增益（率),基尼指数增益，都按照最大值选择特征属性
    2.创建决策树fit(),递归算法实现，注意出口条件
    3.预测predict_proba(),predict(),-->对树的搜索，从根到叶
    4.数据的预处理操作，尤其是连续数据的离散化，分箱
    5.剪枝处理
    '''

    def __init__(self, criterion='cart', is_feature_all_R=False,
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
        self.min_samples_split = min_samples_split  # 最小的划分节点的样本量，小于则不划分
        self.min_sample_leaf = min_samples_leaf  # 叶子节点所包含的最小样本量，剩余的样本小于这个值，标记叶子节点
        self.min_impurity_decrease = min_impurity_decrease  # 最小结点不纯度减少值，小于这个值，不足以划分
        self.max_bins = max_bins  # 连续数据的分箱数，越大，则划分越细
        self.root_node: TreeNode_C() = None  # 分类决策树的根节点
        self.dbw = DataBinWrapper(max_bins=max_bins)  # 连续数据离散化对象
        self.dbw_XrangeMap = {}  # 存储训练样本连续特征分箱的段点
        self.class_values = None  # 样本的类别取值

    def _data_bin_wrapper(self,x_samples):
        '''
        针对特征的连续的特征属性索引dbw_feature_idx,分别进行分箱,
        考虑测试样本与训练样本使用同一个XrangeMap
        @param X_samples: 样本：即可是训练样本,也可以是测试样本
        @return:
        '''
        self.dbw_feature_idx=np.asarray(self.dbw_feature_idx)
        x_sample_prop=[]# 分箱之后的数据
        if not self.dbw_XrangeMap:
            # 为空，即创建决策树前所做的分箱操作
            for i in range(x_samples.shape[1]):
                if i in self.dbw_feature_idx:# 说明当前特征是连续数值
                    self.dbw.fit(x_samples[:,i])
                    self.dbw_XrangeMap[i]=self.dbw.XrangeMap
                    x_sample_prop.append(self.dbw.transform(x_samples[:,i]))
                else:
                    x_sample_prop.append(x_samples[:,i])
        else:
            for i in range(x_samples.shape[1]):
                if i in self.dbw_feature_idx:  # 说明当前特征是连续数值
                    x_sample_prop.append(self.dbw.transform(x_samples[:, i],self.dbw_XrangeMap[i]))
                else:
                    x_sample_prop.append(x_samples[:, i])
        return np.asarray(x_sample_prop).T

    def fit(self, x_train, y_train, sample_weight=None):
        '''
        决策树的创建，递归操作
        @param x_train: 训练样本，ndarray,n*k
        @param y_train: 目标集 ：ndarray,(n,)
        @param sample_weight: 各样本的权重(n,)
        @return:
        '''
        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        self.class_values = np.unique(y_train)  # 样本的类别取值
        n_sample, n_features = x_train.shape  # 训练样本的样本量和特征属性数目
        if sample_weight is None:
            sample_weight = np.asarray([1.0] * n_sample)

        self.root_node = TreeNode_C()
        if self.is_feature_all_R:  # 全部是连续数据
            self.dbw.fit(x_train)
            x_train = self.dbw.transform(x_train)
        elif self.dbw_feature_idx:
            x_train = self._data_bin_wrapper(x_train)
        self._build_tree(1, self.root_node, x_train, y_train, sample_weight)
        # print(x_train)

    def _build_tree(self, cur_depth, cur_node: TreeNode_C, x_train, y_train, sample_weight):
        '''
        递归创建决策树算法,核心算法
        @param cur_depth: 递归划分后的树的深度
        @param cur_node: 递归后的当前根节点
        @param x_train: 递归划分后的训练样本
        @param y_train: 递归划分后的目标集合
        @param sample_weight: 递归划分后的各样本权重
        @return:
        '''
        n_samples, n_features = x_train.shape  # 当前样本子集中的样本量和特征属性数目
        target_dist, weight_dist = {}, {}  # 当前样本子集中的样本量和特征属性数目
        class_labels = np.unique(y_train)  # 当前样本类别分布和权重分布:0-->30%,1-->70%
        for label in class_labels:
            target_dist[label] = len(y_train[y_train == label]) / n_samples
            weight_dist[label] = np.mean(sample_weight[y_train == label])
        cur_node.target_dist = target_dist
        cur_node.weight_dict = weight_dist
        cur_node.n_sample = n_samples
        # 判断停止的条件
        if len(target_dist) <= 1:  # 剩余样本仅包括一个类别，无需划分
            return
        if n_samples < self.min_samples_split:  # 剩余样本量小于最小节点划分标准
            return
        # 达到树的最大深度
        if self.max_depth is not None and cur_depth > self.max_depth:
            return

        # 寻找最佳的特征以及取值
        best_idx, best_index_val, best_criterion_val = None, None, 0.0
        for k in range(n_features):
            for f_val in np.unique(x_train[:, k]):
                feat_k_values = (x_train[:, k] == f_val).astype(int)  # 是当前取值f_val 就是1，否则就是0
                criterion_val = self.criterion_func(feat_k_values, y_train, sample_weight)
                if criterion_val > best_criterion_val:
                    best_criterion_val = criterion_val  # 最佳的划分标准值
                    best_idx, best_index_val = k, f_val  # 当前最佳的特征索引以及取值

        # 递归出口判断
        if best_idx is None:  # 当前属性为空，或者所有样本在所欲属性上取值相同，无法划分
            return
        if best_criterion_val <= self.min_impurity_decrease:  # 小于最小纯度阈值，不划分
            return
        # 满足划分条件，仅当前最佳特征索引和特征取值切分节点，填充树节点信息
        cur_node.feature_idx = best_idx  # 最佳特征所在样本的索引
        cur_node.feature_val = best_index_val  # 最佳特征取值
        cur_node.criterion_val = best_criterion_val  # 最佳特征取值的标准
        # print('当前划分的特征索引：', best_idx, '\t取值：', best_index_val, '\t最佳标准值：', best_criterion_val)
        # print('当前节点的类别分布', target_dist)

        # 创建左子树，并递归创建以当前节点为子树根节点的左子树
        left_index = np.where(x_train[:, best_idx] == best_index_val)  # 左子树所包含的样本子集索引
        if len(left_index[0]) >= self.min_sample_leaf:  # 小于叶子节点所包含的最小样本量，则标记为叶子节点
            left_child_node = TreeNode_C()  # 创建左子树空节点
            cur_node.left_child_node = left_child_node
            # 以当前节点为子树根节点，递归创建
            self._build_tree(cur_depth + 1, left_child_node, x_train[left_index],
                             y_train[left_index], sample_weight[left_index])
        # 创建右子树，并递归创建以右子树为子树根节点的左子树
        right_index = np.where(x_train[:, best_idx] != best_index_val)  # 右子树所包含的样本子集索引
        if len(right_index[0]) >= self.min_sample_leaf:  # 小于叶子节点所包含的最少样本量，则标记为叶子节点
            right_child_Node = TreeNode_C()  # 创建右子树空树节点
            # 以当前节点为右子节点，递归创建
            cur_node.right_child_node = right_child_Node
            self._build_tree(cur_depth + 1, right_child_Node, x_train[right_index],
                             y_train[right_index], sample_weight[right_index])

    def _search_tree_predict(self, cur_node: TreeNode_C, x_test):
        '''
        根据测试样本从根节点到叶子节点搜索路径，判定类别
        搜索：按照后续遍历
        @param cur_node: 单个测试样本
        @param x_test:
        @return:
        '''
        if cur_node.left_child_node and x_test[cur_node.feature_idx] == cur_node.feature_val:
            return self._search_tree_predict(cur_node.left_child_node, x_test)
        elif cur_node.right_child_node and x_test[cur_node.feature_idx] != cur_node.feature_val:
            return self._search_tree_predict(cur_node.right_child_node, x_test)
        else:
            # 叶子节点 :类别，包含有类别分布
            class_p = np.zeros(len(self.class_values))  # 测试样本类别概率
            for i, c in enumerate(self.class_values):
                class_p[i] = cur_node.target_dist.get(c, 0) * cur_node.weight_dict.get(c, 1.0)
            class_p /= np.sum(class_p)  # 归一化
            return class_p

    def predict_proba(self, x_test):
        '''
        预测样本x_test 的类别概率
        @param x_test:  测试样本ndarray,numpy数值运算
        @return:
        '''
        x_test = np.asarray(x_test)  # 避免传DataFrame,List
        if self.is_feature_all_R:
            if self.dbw_XrangeMap is not None:
                x_test = self.dbw.transform(x_test)
            else:
                raise ValueError("请先创建决策树...")
        elif self.dbw_feature_idx is not None:
            x_test = self._data_bin_wrapper(x_test)
        prob_dist = []  # 用于存储测试样本呢的类别概率分布
        for i in range(x_test.shape[0]):
            prob_dist.append(self._search_tree_predict(self.root_node, x_test[i]))
        return np.asarray(prob_dist)

    def predict(self, x_test):
        '''
        预测测试样本的类别
        @param x_test: 测试样本
        @return:
        '''
        x_test = np.asarray(x_test)  # 避免传递DataFrame,list....
        return np.argmax(self.predict_proba(x_test), axis=1)
