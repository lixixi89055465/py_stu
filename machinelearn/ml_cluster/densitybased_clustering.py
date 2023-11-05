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
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
from sklearn.cluster import DBSCAN


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
        self.label_ = None  # 样本所属簇标记列表

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
        DBSCAN密度聚类算法流程
        :return:
        '''
        self.label_ = -1 * np.ones(self.m)  # 初始化样本为噪声，Noise
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
                if q in core_objects:  # 判断是否是核心对象
                    # 如果核心对象q的领域内样本，又有核心对象q，就考察q的领域，
                    # 此时把核心对象p、q以及其邻域内样本都当作一个簇
                    # 接着还要考察q领域内是否还有核心对象？.........
                    #
                    # 获取其邻域内未被访问的样本交集
                    delta = set(np.argwhere(dist_mat[q] <= self.epsilon). \
                                reshape(-1).tolist()) & unvisited_set
                    for d in delta:
                        queue_obj.put(d)  # 领域内未被访问的样本加入队列
                    unvisited_set = unvisited_set - delta  # 从未被访问的集合中减去q领域对象
            # 获取簇聚类idx
            cluster_k = unvisited_set_old - unvisited_set  # 类型是集合
            self.label_[list(cluster_k)] = k  # q以及其领域样本标记为同一个簇
            k += 1  # 用于标记下一个簇，即下一个簇样本的所属簇编号
            core_objects = core_objects - cluster_k  # 去掉以访问的核心对象
        print(self.label_)

    def predict(self):
        '''
        预测样本所属的簇
        :return:
        '''
        self.fit_dbscan()
        return self.label_


if __name__ == '__main__':

    # centers = np.array([
    #     [0.2, 2.3], [-1.5, 2.3], [-2.8, 2], [-2.8, 3], [-2.8, 1]
    # ])
    # std=np.array([0.3,0.2,0.1,0.1,0.1])
    # X,_=make_blobs(n_samples=2000,n_features=2,centers=centers,\
    #                cluster_std=std,random_state=7)
    # dbs=DensityClustering_DBSCAN(X,epsilon=0.2,min_pts=8)
    # dbs.fit_dbscan()
    # labels=dbs.predict()
    # plt.figure(figsize=(7,5))
    # cluster=X[labels==-1]
    # plt.plot(cluster[:,0],cluster[:,1],'ko',label='noise')
    # markers='sp<>*'
    # for i in range(len(np.unique(labels)) - 1):
    #     cluster = X[labels == i]
    #     plt.plot(cluster[:,0], cluster[:,1], markers[i], label='cluster' + str(i + 1))
    # plt.xlabel('Feature 1 ', fontdict={'fontsize': 12})
    # plt.ylabel('Feature 2 ', fontdict={'fontsize': 12})
    # plt.title('Density Based Clustering of DBSCAN Algorithm')
    # plt.show()

    #####################################################
    # X = pd.read_csv('../data/watermelon4.0.csv').values
    # dbs = DensityClustering_DBSCAN(X, epsilon=0.11, min_pts=5)
    # dbs.fit_dbscan()
    # labels = dbs.predict()
    # plt.figure(figsize=(7, 5))
    # cluster = X[labels == -1]  # noise
    # plt.plot(cluster[0], cluster[1], 'ko', label='noise')
    # markers = 'sp<>*'
    # for i in range(len(np.unique(labels)) - 1):
    #     cluster = X[labels == i]
    #     plt.plot(cluster[:,0], cluster[:,1], markers[i], label='cluster' + str(i + 1))
    # plt.xlabel('Feature 1 ', fontdict={'fontsize': 12})
    # plt.ylabel('Feature 2 ', fontdict={'fontsize': 12})
    # plt.title('Density Based Clustering of DBSCAN Algorithm')
    # plt.show()
    ##################################
    X = pd.read_csv('../data/consumption_data.csv')
    X = StandardScaler().fit_transform(X)
    dbs = DensityClustering_DBSCAN(X, epsilon=0.85, min_pts=5)
    # dbs = DBSCAN(eps=0.85, min_samples=5).fit(X)
    dbs.fit_dbscan()
    labels=dbs.predict()
    # labels = dbs.labels_
    title = ['R index', 'F index', 'M index']
    plt.figure(figsize=(7, 10))
    cluster_k = np.unique(labels)
    for f in range(X.shape[1]):  # f表示特征
        plt.subplot(311 + f)
        for c in cluster_k:  # c表示簇索引
            sns.kdeplot(X[labels == c][:, f])
        plt.grid()
        plt.title(title[f])
    plt.show()
