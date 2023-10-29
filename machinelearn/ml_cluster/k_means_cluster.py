# -*- coding: utf-8 -*-
# @Time    : 2023/10/29 10:07
# @Author  : nanji
# @Site    : 
# @File    : k_means_cluster.py
# @Software: PyCharm 
# @Comment : https://www.bilibili.com/video/BV14p4y1h7ay/?p=50&spm_id_from=333.788.top_right_bar_window_history.content.click&vd_source=50305204d8a1be81f31d861b12d4d5cf
import numpy as np
import pandas as pd


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
        random_j = np.random.choice(self.m, 1)  # 随机选择一个簇中心作为样本索引
        self.cluster_centers[0] = self.X[random_j]


if __name__ == '__main__':
    X = pd.read_csv('../data/watermelon4.0.csv')
    kmc = KMeansCluster(X)
    print(kmc.select_cluster_center())
