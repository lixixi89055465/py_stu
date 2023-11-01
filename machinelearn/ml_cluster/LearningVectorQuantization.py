# -*- coding: utf-8 -*-
# @Time    : 2023/10/31 23:36
# @Author  : nanji
# @Site    : 
# @File    : LearningVectorQuantization.py
# @Software: PyCharm 
# @Comment :
import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


class LearningVectorQuantizationClustering:
    '''
    基于原型聚类：学习向量量化
    '''

    def __init__(self, X, y, max_epochs=100, eta=1e-3, tol=1e-3, \
                 dist_method='euclidean'):
        '''
        :param X:
        :param y:
        :param max_epochs:
        :param eta: 学习率
        :param tol: 终止条件
        :param dist_method: 距离函数，默认欧式距离
        '''
        self.m = X.shape[0]
        self.class_label = np.unique(y)
        self.max_epochs = max_epochs
        self.eta = eta
        self.tol = tol
        self.dist_method = dist_method
        self.distance_fun = self.distance_function()  # 距离度量函数
        self.cluster_centers_ = dict()  # 记录簇中心坐标，以类别为键

    def distance_function(self):
        '''
        距离度量函数：euclidean,manhattan,VDM,cos,mahalanobis...
        :return:
        '''
        if self.dist_method == 'euclidean':
            return lambda x, y: np.sqrt(((x - y) ** 2).sum())
        elif self.dist_method == 'manhattan':
            return lambda x, y: np.abs(x - y).sum()
        elif self.dist_method == '':
            return None

    def fit_LVQ(self, X, y):
        '''
        学习向量量化LVQ 核心算法，即根据样本类别，不断更新原型向量
        :param X: 样本数据
        :param y: 目标值
        :return:
        '''
        # 初始化原型向量，随机选择
        random_idx = np.random.choice(self.m, len(self.class_label), replace=False)
        for idx, r_idx in enumerate(random_idx):
            self.cluster_centers_[idx] = X[r_idx]
        for c, idx in enumerate(random_idx):
            idx_s = list(range(self.m))
            np.random.shuffle(idx_s)
            cluster_center_old = copy.deepcopy(self.cluster_centers_)
            eps = 0.
            for idx in idx_s:  # 针对每一个随机样本
                vec, yi = X[idx], y[idx]  # 取某一个样本和类别标记
                min_dist, bst_cid = np.infty, None
                for cid in self.cluster_centers_.keys():
                    dist = self.distance_fun(vec, self.cluster_centers_[cid])
                    if dist < min_dist:
                        min_dist, bst_cid = dist, cid
                # 更新原型向量
                if yi == self.class_label[bst_cid]:
                    self.cluster_centers_[bst_cid] = self.cluster_centers_[bst_cid] \
                                                     + self.eta * (vec - self.cluster_centers_[bst_cid])
                else:
                    self.cluster_centers_[bst_cid] = self.cluster_centers_[bst_cid] \
                                                     - self.eta * (vec - self.cluster_centers_[bst_cid])
            # 终止条件
            for key in self.cluster_centers_.keys():
                eps += self.distance_fun(cluster_center_old[key] , self.cluster_centers_[key])
            eps /= len(self.cluster_centers_)
            if eps < self.tol:
                break

        print(random_idx)

    def predict(self, X):
        cluster_labels = []
        for i in range(self.m):
            best_k, min_dist = np.infty
            for idx in range(len(self.cluster_centers_)):
                dist = self.distance_fun(self.cluster_centers_[idx], X[i])
                if dist < min_dist:
                    best_k, min_dist = idx, dist
            cluster_labels.append(best_k)
        return cluster_labels


if __name__ == '__main__':
    data = pd.read_csv('../data/iris.csv')
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1]
    y = LabelEncoder().fit_transform(y)
    lvq = LearningVectorQuantizationClustering(X, y, eta=0.05, tol=1e-3)
    lvq.fit_LVQ(X, y)
    cluster_ind = lvq.predict(X)
    print(classification_report(y, cluster_ind))
