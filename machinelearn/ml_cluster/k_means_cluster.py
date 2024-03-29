# -*- coding: utf-8 -*-
# @Time    : 2023/10/29 10:07
# @Author  : nanji
# @Site    : 
# @File    : k_means_cluster.py
# @Software: PyCharm 
# @Comment : https://www.bilibili.com/video/BV14p4y1h7ay/?p=50&spm_id_from=333.788.top_right_bar_window_history.content.click&vd_source=50305204d8a1be81f31d861b12d4d5cf
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import seaborn as sns

import matplotlib.pyplot as plt


class KMeansCluster:
    '''
    基于原型聚类的，K-均值聚类
    '''

    def __init__(self, data, k=3, max_epochs=100, tol=13 - 3, \
                 dist_method='euclidean'):
        '''
        :param k: 聚类簇族
        :param max_epochs: 最大迭代次数
        :param tol: 精度要求，即迭代停止条件
        :param dist_method:  距离度量方法，默认按'欧氏距离‘计算
        '''
        self.X = data
        self.k = k
        self.m = data.shape[0]
        self.max_epochs = max_epochs
        self.tol = tol
        self.dist_method = dist_method
        self.distance_fun = self.distance_function()  # 距离度量函数
        self.cluster_centers = dict()  # 存储簇中心向量

    def distance_function(self):
        '''
        距离度量函数：euclidean,manhattan,VDM,cos,mahalanobis...
        :return:
        '''
        if self.dist_method == 'euclidean':
            return lambda x, y: np.sqrt(((x - y) ** 2).sum())
        elif self.dist_method == '':
            return None

    def select_cluster_center(self):
        '''
        按照k-means++方法，初始化簇中心向量
        :return:
        '''
        sample_j = np.random.choice(self.m, 1)  # 随机选择一个簇中心作为样本索引
        self.cluster_centers[0] = self.X[sample_j]
        select_center_vec = [sample_j]  # 以选择的簇中心样本索引存储，防止在被宣导
        while len(self.cluster_centers) < self.k:
            sample_j, max_dist = None, 0
            for j in range(self.m):
                for key in self.cluster_centers.keys():
                    # 计算当前样本距离每个簇中心的距离
                    dist = self.distance_fun(self.cluster_centers[key], self.X[j])
                    if dist > max_dist and j not in select_center_vec:
                        sample_j, max_dist = j, dist
            select_center_vec.append(sample_j)

            self.cluster_centers[len(self.cluster_centers)] = self.X[sample_j]
        print('k-means++算法，初始化簇中心向量为：')
        for key in self.cluster_centers.keys():
            print('簇' + str(key + 1), self.cluster_centers[key])
        print('-' * 100)

    def fit_kmeans(self):
        '''
        k均值算法的核心内容，实质就是更新簇中心向量
        :return:
        '''
        for epoch in range(self.max_epochs):
            cluster = dict()  #
            for idx in range(self.k):
                cluster[idx] = []
            for j in range(self.m):
                best_k, min_dist = None, np.infty
                for idx in self.cluster_centers.keys():
                    dist = self.distance_fun(self.cluster_centers[idx], self.X[j])
                    if dist < min_dist:
                        best_k, min_dist = idx, dist  # 取最近的距离
                cluster[best_k].append(j)
            # 更新簇中心均值向量
            eps = 0  # 更新卡后的中心点的差距
            for c_idx in cluster.keys():
                vec_k = np.mean(self.X[cluster[c_idx]], axis=0)
                # 各簇内距离之和
                eps += self.distance_fun(vec_k, self.cluster_centers[idx])
                self.cluster_centers[c_idx] = vec_k  # 更新簇中心

            # 簇中心更新过程的输出
            print('iter', epoch + 1, ':', '簇中心与簇内样本索引：')
            for key in cluster.keys():
                print('簇' + str(key + 1), ',中心', self.cluster_centers[key], \
                      '样本索引', cluster[key])
            print('-' * 100)
            # 判断终止迭代的条件
            if eps < self.tol:
                break

    def predict(self, X):
        '''
        针对每个样本，根据各个簇中心计算距离，距离哪一个中心近，归于那个簇
        :param X: 预测样本数据
        :return:
        '''
        cluster_labels = []  # 簇中心索引
        for i in range(X.shape[0]):
            best_j, min_dist = None, np.infty
            for idx in range(self.k):
                dist = self.distance_fun(self.cluster_centers[idx], X[i])
                if dist < min_dist:
                    min_dist, best_j = dist, idx
            cluster_labels.append(best_j)
        return np.asarray(cluster_labels)

    def plt_clasify(self):
        '''
        绘制分类结果图，并绘制分类边界
        :param model:
        :return:
        '''
        # 绘制分类边界，模拟数据，生成网络点并预测，pcolormesh绘制
        x1_min, x2_min = X.min(axis=0)
        x1_max, x2_max = X.max(axis=0)
        t1 = np.linspace(x1_min, x1_max, 50)
        t2 = np.linspace(x2_min, x2_max, 50)
        x1, x2 = np.meshgrid(t1, t2)  # 生成网络采样点50*50
        x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点2500*2
        # cm_light = ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
        # cm_dark = ListedColormap(['darkgreen', 'darkred', 'darkblue'])
        cm_light = ListedColormap(['g', 'r', 'b', 'm', 'c'])
        cm_dark = ListedColormap(['g', 'r', 'b', 'm', 'c'])

        y_show_hat = self.predict(x_show)
        y_show_hat = y_show_hat.reshape(x1.shape)

        plt.figure(facecolor='w')
        plt.pcolormesh(x1, x2, y_show_hat, shading='auto', cmap=cm_light, alpha=0.3)
        # plt.scatter(self.X[:, 0], self.X[:, 1], c=self.predict(self.X).ravel(), \
        #             edgecolors='k', s=20, cmap=cm_dark)  # 全部数据
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.predict(self.X).ravel(), \
                    s=20, cmap=cm_dark)  # 全部数据
        for key in self.cluster_centers.keys():
            center = self.cluster_centers[key]
            plt.scatter(center[0], center[1], c='k', marker='p', s=100)

        plt.xlabel('X1', fontsize=11)
        plt.ylabel('X2', fontsize=11)
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)

        plt.grid(b=True, ls=':', color='#606060')
        plt.title('K-means classification boundary and Cluster Center Vec', fontsize=12)
        plt.show()


if __name__ == '__main__':
    # X = pd.read_csv('../data/watermelon4.0.csv').values
    # kmc = KMeansCluster(X, k=3, tol=1e-8)
    # kmc.select_cluster_center()
    # kmc.fit_kmeans()
    # labels = kmc.predict(X)
    # print(labels)
    # for key in kmc.cluster_centers.keys():
    #     print('簇' + str(key + 1), kmc.cluster_centers[key])
    # kmc.plt_clasify()
    ###############################################
    # centers = np.array([
    #     [0.2, 2.3], [-1.5, 2.3], [-2.8, 1.8], [-2.8, 2.8], [-2.8, 1.3]
    # ])
    # std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    # X, _ = make_blobs(n_samples=2000, n_features=2, centers=centers, cluster_std=std, random_state=7)
    #
    # kmc = KMeansCluster(X, k=5, tol=1e-8)
    # kmc.select_cluster_center()
    # kmc.fit_kmeans()
    # labels = kmc.predict(X)
    # print(labels)
    # for key in kmc.cluster_centers.keys():
    #     print('簇' + str(key + 1), kmc.cluster_centers[key])
    # kmc.plt_clasify()
    ###############################################
    X = pd.read_csv('../data/consumption_data.csv').values
    X_scaler = StandardScaler().fit_transform(X)
    cluster_k = 3
    kmc = KMeansCluster(X_scaler, k=3, tol=1e-8)
    kmc.select_cluster_center()
    kmc.fit_kmeans()
    labels = kmc.predict(X_scaler)
    for key in kmc.cluster_centers.keys():
        print('簇' + str(key + 1), kmc.cluster_centers[key])
    # kmc.plt_clasify()
    #  可视化核密度估计
    title = ['R index', 'F index', 'M index']
    plt.figure(figsize=(7, 10))
    for f in range(X.shape[1]):  # f表示特征
        plt.subplot(311 + f)
        for c in range(cluster_k):  # c表示簇索引
            sns.kdeplot(X[labels == c][:, f])
        plt.grid()
        plt.title(title[f])
    plt.show()
    # Sklearn.kmeans
    from sklearn.cluster import KMeans

    skm = KMeans(n_clusters=cluster_k).fit(X)
    print(skm.cluster_centers_)
    title = ['SR index', 'SF index', 'SM index']
    plt.figure(figsize=(7, 10))
    for f in range(X.shape[1]):  # f表示特征
        plt.subplot(311 + f)
        for c in range(cluster_k):  # c 表示簇索引
            sns.kdeplot(X[skm.labels_ == c][:, f])
        plt.grid()
        plt.title(title[f])
    plt.show()
