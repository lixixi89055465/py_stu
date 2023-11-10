# -*- coding: utf-8 -*-
# @Time    : 2023/11/5 15:46
# @Author  : nanji
# @Site    : 
# @File    : hierarchicalclustering.py
# @Software: PyCharm 
# @Comment :
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns


class HierarchicalClustering_AGNES:
    '''
    基于AGNES凝聚的层次聚类算法　

    '''

    def __init__(self, X, k=3, dist_samples='euclidean', \
                 dist_cluster='average_linkage'):
        '''
        层次聚类算法类对象初始化
        :param X:  样本
        :param k:  聚类聚簇
        :param dist_samples: 样本距离
        :param dist_cluster: 簇间距离
        '''
        self.X = X
        self.m = X.shape[0]
        self.k = k
        self.dist_samples = dist_samples
        self.dist_cluster = dist_cluster
        self.G = None  # 簇
        self.cluster_center_ = dict()  # 簇中心点向导

    def distance_samples(self):
        '''
        初始化样本间距离度量 :
        :return:
        '''
        dist_mat = np.zeros(shape=(self.m, self.m))  # m*m
        if self.dist_samples == 'euclidean':
            # 欧式距离
            for i in range(0, self.m - 1):
                for j in range(i + 1, self.m):
                    dist_mat[i][j] = np.sqrt(((self.X[i] - self.X[j]) ** 2).sum())
                    dist_mat[j][i] = dist_mat[i][j]
        elif self.dist_samples == 'mahalanobis':
            # 马氏距离
            sigma = np.linalg.inv(np.cov(self.X.T))  # 样本协方差矩阵
            for i in range(0, self.m - 1):
                for j in range(i + 1, self.m):
                    X_ij = self.X[i] - self.X[j]
                    dist_mat[i, j] = np.sqrt(X_ij.dot(np.linalg.inv(sigma).dot(X_ij)))
                    dist_mat[j, i] = dist_mat[i, j]
        return dist_mat

    def distance_cluster(self, C_i, C_j):
        '''
        簇间距离度量，single_linage,complete_linkage,average_linkage
        :param C_i: 第i个簇样本集合
        :param C_j: 第j个簇样本集合
        :return:
        '''
        if self.dist_cluster == 'average_linkage':
            return np.sqrt(((np.mean(C_i, axis=0) - np.mean(C_j, axis=0)) ** 2).sum())
        elif self.dist_cluster == 'single_linkage':
            min_dist = 0
            for i in range(len(C_i)):
                for j in range(len(C_j)):
                    dist = np.sqrt(((C_i[i] - C_j[j]) ** 2).sum())
                    if dist < min_dist:
                        min_dist = dist
            return min_dist
        elif self.dist_cluster == 'complete_linkage':
            max_dist = 0
            for i in range(len(C_i)):
                for j in range(len(C_j)):
                    dist = np.sqrt(((C_i[i] - C_j[j]) ** 2).sum())
                    if dist > max_dist:
                        max_dist = dist
            return max_dist
        else:
            return None

    def fit_agnes(self):
        '''
        层次聚类核心算法，它先将数据集中的每个样本看作一个初始聚类簇
        然后在算法运行的每一步中找出距离最近的两个聚类簇进行合并，
        该过程不断重复，直至达到预设的聚类簇个数。
        :return:
        '''
        self.C = dict()  # 初始化簇，即每个样本自成一个簇
        for i in range(self.m):
            self.C[i] = self.X[i]
        # 初始化样本间距离计算
        dist_mat = self.distance_samples()
        q = self.m  # 初始时，每个样本自称一个簇，共m个
        while q > self.k:
            # 查找最近的连个簇Ci,Cj，然后进行合并
            min_dist, i_, j_ = np.infty, None, None
            for i in range(q - 1):
                for j in range(i + 1, q):
                    if dist_mat[i, j] < min_dist:
                        min_dist, i_, j_ = dist_mat[i, j], i, j
            # 合并两个最近的簇到Ci*
            self.C[i_] = np.concatenate([self.C[i_], self.C[j_]])
            # 将聚类簇Ci重编号为 Cj-1
            for j in range(j_ + 1, q):
                self.C[j - 1] = self.C[j]
            # print(self.C)
            del self.C[q - 1]  # 删除，即最后一个
            # 删除距离矩阵第j*行和第j*列
            dist_mat = np.delete(dist_mat, j_, axis=0)  # 列
            dist_mat = np.delete(dist_mat, j_, axis=1)  # 行
            for j in range(q - 1):
                dist_mat[i_, j] = self.distance_cluster(self.C[i_], self.C[j])
                dist_mat[j, i_] = dist_mat[i_, j]
            q -= 1  # 每次合并距离最近的两个簇
        # 满足聚类簇数k要求后，求解各个簇的中心点向量，用于预测
        for key in self.C.keys():
            self.cluster_center_[key] = np.mean(self.C[key], axis=0)
        # print(self.cluster_center_)

    def predict(self):
        '''
        根据各簇中心向量，采用欧式距离，预测样本所属簇标记
        :return:
        '''
        cluster_labels = []
        for row in range(self.m):
            min_dist, best_label = np.infty, None
            for c_num in self.cluster_center_.keys():
                dist = np.sqrt(((self.X[row] - self.cluster_center_[c_num]) ** 2).sum())
                if dist < min_dist:
                    min_dist, best_label = dist, c_num
            cluster_labels.append(best_label)
        return np.array(cluster_labels)


if __name__ == '__main__':
    # X = pd.read_csv('../data/watermelon4.0.csv').values
    # # hc = HierarchicalClustering_AGNES(X[:5, :], dist_samples='mahalanobis')
    # hc = HierarchicalClustering_AGNES(X, dist_samples='mahalanobis')
    # # hc.distance_samples()
    # hc.fit_agnes()
    #############################################
    # centers = np.array([[0.2, 2.3], [-1, 2.3], [-3, 2], [-2, 2.8], [1, 3]])
    # std = np.array([0.3, 0.2, 0.15, 0.15, 0.2])
    # X, _ = make_blobs(n_samples=500, n_features=2, \
    #                   centers=centers, cluster_std=std, random_state=7)
    # hc = HierarchicalClustering_AGNES(X, k=5, dist_samples='euclidean')
    # hc.fit_agnes()
    # labels = hc.predict()
    # print(labels)
    # print('=' * 100)
    # plt.figure(figsize=(7, 5))
    # markers = 'osp<>*'
    # for i in range(len(np.unique(labels))):
    #     cluster = X[labels == i]
    #     plt.plot(cluster[:, 0], cluster[:, 1], markers[i], label='cluster' + str(i + 1))
    # plt.xlabel('Feature 1 ', fontdict={'fontsize': 12})
    # plt.ylabel('Feature 2 ', fontdict={'fontsize': 12})
    # plt.title('HierarchicalClustering_AGNES Clustering of DBSCAN Algorithm')
    # plt.legend()
    # plt.grid(ls=':')
    # plt.savefig('2.png')
    # plt.show()

    ######################################
    X = pd.read_csv('../data/consumption_data.csv')
    X = StandardScaler().fit_transform(X)
    hc = HierarchicalClustering_AGNES(X, k=3, dist_cluster='complete_linkage')
    hc.fit_agnes()
    labels = hc.predict()

    title = ['R index', 'F index', 'M index']
    plt.figure(figsize=(7, 10))
    cluster_k = np.unique(labels)
    for f in range(X.shape[1]):
        plt.subplot(311 + f)
        for c in cluster_k:
            sns.kdeplot(X[labels == c][:, f])
        plt.grid()
        plt.title(title[f])
    plt.savefig('hierarchicalclustering.py.png')
    plt.show()
