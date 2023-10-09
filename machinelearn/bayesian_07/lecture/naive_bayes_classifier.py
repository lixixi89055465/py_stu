# -*- coding: utf-8 -*-
# @Time    : 2023/10/8 下午11:22
# @Author  : nanji
# @Site    : 
# @File    : naive_bayes_classifier.py
# @Software: PyCharm 
# @Comment :
import numpy as np  #
import collections as cc  # 集合的技术功能
from scipy.stats import norm  # 极大似然估计
from machinelearn.decision_tree_04.utils.data_bin_wrapper import DataBinWrapper


class NaiveBayesClassifier:
    '''
    朴素贝叶斯分类起：对于连续属性两种方式操作，1是分箱处理，2是
    直接进行高斯分布的参数估计
    '''

    def __init__(self, is_binned=False, is_feature_all_R=False, \
                 feature_R_idx=None, max_bins=10):
        self.is_binned = is_binned  # 连续特征变量数据是否进行分箱操作，离散化
        if is_binned:
            self.is_feature_all_R = is_feature_all_R  # 是否所有特征变量都是连续树脂 bool
            self.max_bins = max_bins  # 最大分箱数
            self.dbw = DataBinWrapper()  # 分箱对象
            self.dbw_XrangeMap = dict()  # 存储训练样本特征分箱的端点
        self.feature_R_idx = feature_R_idx  # 混合式数据中连续特征变量的索引
        self.class_values, self.n_class = None, 0  # 类别取值以及类别数
        self.prior_prob = dict()  # 先验分布，键是类别取值，值是先验概率
        self.classified_feature_prob = dict()  # 存储每个类所对应的特征变量取值频次或连续属性的高斯分布参数
        self.class_values_num = dict()  # 目标集中每个类别的样本量:Dc
        self.feature_values_num = dict()

    def fit(self, x_train, y_train):
        '''
        朴素贝叶斯分类起训练,将朴素贝叶斯分类起设计的所有概率估值事先计算好存储起来
        :param x_train: 训练集
        :param y_train: 目标集
        :return:
        '''
        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        self.class_values = np.unique(y_train)  # 类别取值
        self.n_class = len(self.class_values)  # 类别数
        if self.n_class < 2:
            print("既有一个类别，不进行贝叶斯分离器估计...")
            exit(0)
        self._prior_probability(y_train)
        # 每个特征变量不同的取值数，类条件概率的分母
        for i in range(x_train.shape[1]):
            self.feature_values_num[i] = len(np.unique(x_train[:, i]))

        if self.is_binned:
            self._binned_fit(x_train, y_train)  # 分箱处理
        else:
            self._gaussian_fit(x_train, y_train)  # 直接进行高斯分布估计

    def _prior_probability(self, y_train):
        '''

        :param y_train:
        :return:
        '''
        n_samples = len(y_train)
        self.class_values_num = cc.Counter(y_train)  # {"否":9,"是":8}
        print(self.class_values_num)
        for key in self.class_values_num.keys():
            self.prior_prob[key] = (self.class_values_num[key] + 1) / (n_samples + self.n_class)
        print(self.prior_prob)

    def _binned_fit(self, x_train, y_train):
        '''
        对连续特征属性进行分箱操作，然后计算各概率值
        :param x_train:
        :param y_train:
        :return:
        '''
        self.dbw.fit(x_train)
        self.dbw.transform(x_train)
        pass

    def _gaussian_fit(self, x_train, y_train):
        '''
        连续特征变量不进行分箱，直接进行高斯分布估计，离散特征变量取值除外
        :param x_train:
        :param y_train:
        :return:
        '''
        for c in self.class_values:
            class_x = x_train[c == y_train]  # 获取对应羸惫的样本
            feature_counter = dict()  # 每个离散变量特征中特定值出现的频次
            # 连续特征变量村u,sigma
            for i in range(x_train.shape[1]):
                if self.feature_R_idx is not None and (i in self.feature_R_idx):  # 连续特征
                    # 极大似然估计均值和方差
                    mu, sigma = norm.fit(x_train[:, i])
                    feature_counter[i] = {'mu': mu, 'sigma': sigma}
                else:  # 离散特征
                    feature_counter[i] = cc.Counter(class_x[:, i])
            self.classified_feature_prob[c] = feature_counter
