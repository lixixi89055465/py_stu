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
    分类决策树算法实现:
    1.划分标准：信息增益（率),基尼指数增益，都按照最大值选择特征属性
    2.创建决策树fit(),递归算法实现，注意出口条件
    3.预测predict_proba(),predict(),-->对树的搜索，从根到叶
    4.数据的预处理操作，尤其是连续数据的离散化，分箱
    5.剪枝处理
    '''

    def __init__(self, criterion='CART', is_feature_all_R=False,
                 dbw_feature_idx=None, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_impurity_decrease=0, max_bins=10,dbw_XrangeMap=None):
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
        self.dbw_XrangeMap=dbw_XrangeMap

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
        x_train,y_train=np.asarray(x_train),np.asarray(y_train)
        n_sample, n_features = x_train.shape  # 训练样本的样本量和特征属性数目
        if sample_weight is None:
            sample_weight = np.asarray([1.0] * n_sample)

        self.root_node = TreeNode_C()
        if self.is_feature_all_R:# 全部是连续数据
            self.dbw.fit(x_train)
            x_train = self.dbw.transform(x_train)
        elif self.dbw_feature_idx:
            x_train=self._data_bin_wrapper(x_train)
        self._build_tree(1, self.root_node, x_train, y_train, sample_weight)
        print(x_train)



    def _build_tree(self, cur_depth, cur_node:TreeNode_C, x_train, y_train, sample_weight):
        '''
        递归创建决策树算法,核心算法
        @param cur_depth:
        @param cur_node:
        @param x_train:
        @param y_train:
        @param sample_weight:
        @return:
        '''

