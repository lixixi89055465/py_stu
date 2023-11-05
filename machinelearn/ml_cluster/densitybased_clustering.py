# -*- coding: utf-8 -*-
# @Time    : 2023/11/4 20:51
# @Author  : nanji
# @Site    : 
# @File    : densitybased_clustering.py
# @Software: PyCharm
# @Comment :
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from queue import Queue  # 队列，特点：先进先出


class DensityClustering_DBSCAN:
    '''
    DBSCAN密度聚类算法
    '''

    def __init__(self, X, epsilon=0.5, min_pts=3, dist_method='euclidean'):
        '''
        基于DBSCAN算法，实现密度聚类
        :param X: 样本数据
        :param epsilon: 邻域
        :param min_pts: 临域内至少包含的样本量
        :param dist_method: 距离度量方法
        '''
        self.X = X
        self.m = X.shape[0]  # 样本量
        self.epsilon = epsilon
        self.min_pts = min_pts
        self.dist_method = dist_method
        self.distance_fun = self.distance_function()  # 存储簇中心向量，以簇数为键

    def distance_function(self):
        '''
        距离度量函数:euclidean,manhattan,VDM,cos,mahalanobis ...
        :return:
        '''
        if self.dist_method == 'euclidean':
            return lambda x, y: np.sqrt(((x - y) ** 2).sum())
        elif self.dist_method == '':
            return None

    def fit_dbscan(self):
        '''
        DBSCAN密度聚类算法 流程
        :return:
        '''
        dist_mat = np.zeros(shape=(self.m, self.m))  # 距离矩阵,m*m
        # 计算样本之间的距离
        for i in range(self.m - 1):
            for j in range(i + 1, self.m):
                dist_mat[i, j] = self.distance_fun(self.X[i], self.X[j])
                dist_mat[j, i] = dist_mat[i, j]
        # 初始化核心对象集合
        core_objects = set()  # 空集合
        for i in range(self.m):
            # 对每一个样本，考察其邻域内样本量是否大于等于minpts
            if np.sum(dist_mat[i] <= self.epsilon) >= self.min_pts:
                core_objects.add(i)  # 添加样本i为核心对象
        k = 0  # 初始化簇数，即簇标记
        unvisited_set = set(range(self.m))  # 初始化未访问的样本集合
        while len(core_objects) > 0:
            unvisited_set_old = unvisited_set.copy()  # 记录当前未访问的样本集合
            # 随机选取一个核心对象
            obj_idx = np.random.choice(list(core_objects))
            queue_obj = Queue()  # 初始化一个队列
            queue_obj.put(obj_idx)  # 核心对象入队列
            # 未访问集合中删除核心对象
            unvisited_set = unvisited_set - {obj_idx}
            while not queue_obj.empty():
                q = queue_obj.get()  # 取队首元素


if __name__ == '__main__':
    X = pd.read_csv('../data/watermelon4.0.csv').values
    dbs = DensityClustering_DBSCAN(X, epsilon=0.2, min_pts=8)
    dbs.fit_dbscan()
