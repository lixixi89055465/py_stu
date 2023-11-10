# -*- coding: utf-8 -*-
# @Time    : 2023/9/23 上午8:10
# @Author  : nanji
# @Site    : 
# @File    : lda2classify.py
# @Software: PyCharm 
# @Comment :

import numpy as np


class LDABinaryClassifier:
    '''
    线性判别分析，而分类模型
    '''

    def __init__(self):
        self.mu = None  # 各類別均值向量
        self.Sw_i = None  # 各类内散度矩阵
        self.Sw = None  # 类内散度矩阵
        self.weight = None  # 模型的系数，投影方向
        self.w0 = None  # 阈值

    def fit(self, x_train, y_train):
        '''
        线性判别分析核心算法,计算投影方向及判别阈值
        :param x_train:
        :param y_train:
        :return:
        '''
        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        class_values = np.sort(np.unique(y_train))  # 不同的类别取值
        n_samples, n_feature = x_train.shape  # 样本量和特征变量数目
        class_size = []  # 计算各类别的样本量
        if len(class_values) != 2:
            raise ValueError("仅限于而分类且线性可分数据 ")
        # 1.计算类均值后，Sw散度矩阵，Sb散度矩阵
        self.Sw_i = dict()  # 字典形式，以类别取值为键，值是对应的类别样本的类内散度
        self.mu = dict()  # 字典形式，以类别取值为键，值是对应的类别样本的均值向量
        self.Sw = np.zeros((n_feature, n_feature))
        for label_val in class_values:
            class_x = x_train[y_train == label_val]  # 按类别对样本进行划分
            class_size.append(len(class_x))  # 该类别的样本量
            self.mu[label_val] = np.mean(class_x, axis=0)  # 对特征取均值构成均值向量
            self.Sw_i[label_val] = (class_x - self.mu[label_val]).T.dot(class_x - self.mu[label_val])
            self.Sw += self.Sw_i[label_val]  # 累加计算类内散度矩阵

        # 2.计算投影方向
        inv_sw = np.linalg.inv(self.Sw)
        self.weight = inv_sw.dot(self.mu[0] - self.mu[1])  # 投影方向

        # 3.计算阈值w0
        self.w0 = (class_size[0] * self.weight.dot(self.mu[0]) +
                   class_size[1] * self.weight.dot(self.mu[1])) / n_samples
        print(self.w0)
        return self.weight

    def predict(self, x_test):
        '''
        根据测试样本进行预测
        :param x_test: 测试样本
        :return:
        '''
        x_test = np.asarray(x_test)
        y_pred = self.weight.dot(x_test.T)
        y_test_pred = np.zeros(x_test.shape[0], dtype=np.int)  # 初始测试样本的类别值
        y_test_pred[y_pred < self.w0] = 1  # 小于阈值的为负值
        return y_test_pred
