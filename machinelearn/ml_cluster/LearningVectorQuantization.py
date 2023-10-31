# -*- coding: utf-8 -*-
# @Time    : 2023/10/31 23:36
# @Author  : nanji
# @Site    : 
# @File    : LearningVectorQuantization.py
# @Software: PyCharm 
# @Comment :
import numpy as np


class LearningVectorQuantization:
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
