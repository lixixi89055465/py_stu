# -*- coding: utf-8 -*-
# @Time    : 2023/10/2 15:38
# @Author  : nanji
# @Site    : 
# @File    : knn_kdtree.py
# @Software: PyCharm 
# @Comment :
from machinelearn.knn_05.distUtils import DistanceUtils
from machinelearn.knn_05.kdtree_node import KDTreeNode
import numpy as np
import heapq
from collections import Counter  # 集合中的计数功能


class KNearestNeighborKDTree:
    '''
    K近邻算法的实现，基于KD树结构
    1.fit:特征向量空间的划分，即构建KD树（建立KNN算法模型)
    2.predict:预测，近邻搜索
    3.可视化kd树
    '''

    def __init__(self, k: int = 5, p=2, view_kdt=False):
        '''
        KNN算法的初始化必要参数
        :param k: 近邻数
        :param p: 距离度量标准
        :param view_kdt: 是否可视化KD树
        '''
        self.k = k  # 预测，紧邻搜索树，使用的参数，表示近邻
        self.p = p  # 预测，近邻搜索时，使用的参数，表示样本的近似度
        self.view_kdt = view_kdt
        self.dist_utils = DistanceUtils(self.p)  # 距离度量的类对象
        self.kdt_root: KDTreeNode() = None  # KD 树的根节点
        self.k_dimension = 0  # 特征空间维度，即样本的特征属性数
        self.k_neighbors = []  # 用于记录某个测试样本的近邻实例点

    def fit(self, x_train, y_train):
        '''
        递归创建KD 树，即对特征向量空间进行划分
        :param x_train:
        :param y_train:
        :return:
        '''
        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        self.k_dimension = x_train.shape[1]  # 特征维度
        idx_array = np.arange(x_train.shape[0])  # 训练样本的索引编号
        self.kdt_root = self._build_kd_tree(x_train, y_train, idx_array, 0)
        if self.view_kdt:
            self.draw_kd_tree()  # 可视化kd树

    def _build_kd_tree(self, x_train, y_train, idx_array, depth):
        '''
        递归创建KD 树，KD树二叉树，严格区分左子树右子树，表示对K维空间的一个划分
        :param x_train:
        :param y_train:
        :param idx_array:
        :param depth:
        :return:
        '''
        if x_train.shape[0] == 0:  # 递归出口
            return
        split_dimension = depth % self.k_dimension  # 数据的划分维度x^(i)
        sorted(x_train, key=lambda x: x[split_dimension])  # 按照某个划分维度排序
        median_idx = x_train.shape[0] // 2  # 中位数所对应的数据的索引
        median_node = x_train[median_idx]  # 切分点作为当前子树的根节点
        # 划分左右子树区域
        left_instances, right_instances = x_train[:median_idx], x_train[median_idx + 1:]
        left_labels, right_labels = y_train[:median_idx], y_train[median_idx + 1:]
        left_idx, right_idx = idx_array[:median_idx], idx_array[median_idx + 1:]

        # 递归调用
        left_child = self._build_kd_tree(left_instances, left_labels, left_idx, depth + 1)
        right_child = self._build_kd_tree(right_instances, right_labels, right_idx, depth + 1)
        kdt_new_node = KDTreeNode(median_node, y_train[median_idx], idx_array[median_idx],
                                  split_dimension, left_child, right_child, depth)
        return kdt_new_node

    def predict(self, x_test):
        '''
        KD树的近邻搜索，即测试样本的预测
        :param x_test: 测试样本，ndarray:(n*k)
        :return:
        '''
        x_test = np.asarray(x_test)
        if self.kdt_root is None:
            raise ValueError('KDTree is None, Please fitKDTree...')
        elif x_test.shape[1] != self.k_dimension:
            raise ValueError("Test Samples dimension unmatched KDTree's dimension.")
        else:
            y_test_hat = []  # 用于存储测试样本的预测类别
            for i in range(x_test.shape[0]):
                self.k_neighbors = []
                self._search_kd_tree(self.kdt_root, x_test[i])
                y_test_labels = []
                # 取每个近邻样本的类别标签
                for k in range(self.k):
                    y_test_labels.append(self.k_neighbors[k]['label'])
                # 按分类规则，（多数表决法)
                counter = Counter(y_test_labels)
                idx = np.argmax(list(counter.values()))
                y_test_hat.append(list(counter.keys())[idx])
        return np.asarray(y_test_hat)

    def draw_kd_tree(self):
        '''
        可视化 kd树
        :return:
        '''

    def _search_kd_tree(self, kd_tree: KDTreeNode, x_test):
        '''
        kd树的递归搜索，即测试样本的预测
        数据结构：堆排序，搜索过程中，维护一个小根堆
        :param kd_tree:已构建的KD树
        :param x_test: 单个测试样本
        :return:
        '''
        if kd_tree is None:  # 递归出口
            return
            # 计算测试样本与当前KD子树的根结点的距离（相似度）

        distance = self.dist_utils.distance_func(kd_tree.instance_node, x_test)
        # 1.如果不够k个样本，继续递归
        # 2.如果搜索了k个样本，但是k个样本未必是最近邻的。
        # 当前计算的实例点的距离小于k个样本的最大距离，则递归，大于最大距离，没必要递归
        if (len(self.k_neighbors) < self.k) or (distance < self.k_neighbors[-1]['distance']):
            self._search_kd_tree(kd_tree.left_child, x_test)  # 递归左子树
            self._search_kd_tree(kd_tree.right_child, x_test)  # 递归右子树
            # 在整个搜索路径上的kd树的节点，存储在self.k_neighbors中，包含三个值
            # 当前实例点，类别，距离
            self.k_neighbors.append({
                "node": kd_tree.instance_node,  # 结点
                "label": kd_tree.instance_label,  # 当前实例的类别
                "distance": distance  # 当前实例点与测试样本呢的距离
            })
            # 按照距离进行排序，选择最小的k个最近邻样本实例,更新最近邻距离
            # 小根堆，k_neighbors中第一个结点是距离测试样本最近的
            self.k_neighbors = heapq.nsmallest(self.k, self.k_neighbors, key=lambda d: d['distance'])

        pass
