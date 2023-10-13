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
        self.is_feature_all_R = False
        self.dbw_XrangeMap = None
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

    def _data_bin_wrapper(self, x_samples):
        '''
        针对特征的连续的特征属性索引dbw_idx,分别进行分箱,
        考虑测试样本与训练样本使用同一个XrangeMap
        @param X_samples: 样本：即可是训练样本,也可以是测试样本
        @return:
        '''
        self.feature_R_idx = np.asarray(self.feature_R_idx)
        x_sample_prop = []  # 分箱之后的数据
        if not self.dbw_XrangeMap:
            self.dbw_XrangeMap = dict()
            self.dbw = DataBinWrapper()
            # 为空，即创建决策树前所做的分箱操作
            for i in range(x_samples.shape[1]):
                if i in self.feature_R_idx:  # 说明当前特征是连续数值
                    self.dbw.fit(x_samples[:, i])
                    self.dbw_XrangeMap[i] = self.dbw.XrangeMap
                    x_sample_prop.append(self.dbw.transform(x_samples[:, i]))
                else:
                    x_sample_prop.append(x_samples[:, i])
        else:
            for i in range(x_samples.shape[1]):
                if i in self.feature_R_idx:  # 说明当前特征是连续数值
                    x_sample_prop.append(self.dbw.transform(x_samples[:, i], self.dbw_XrangeMap[i]))
                else:
                    x_sample_prop.append(x_samples[:, i])
        return np.asarray(x_sample_prop).T

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
        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        if self.is_feature_all_R:  # 全部是连续
            self.dbw.fit(x_train)
            x_train = self.dbw.transform(x_train)
        elif self.feature_R_idx is not None:
            x_train = self._data_bin_wrapper(x_train)
        for c in self.class_values:
            class_x = x_train[c == y_train]  # 获取对应羸惫的样本
            feature_counter = dict()  # 每个离散变量特征中特定值出现的频次
            # 连续特征变量村u,sigma
            for i in range(x_train.shape[1]):
                feature_counter[i] = cc.Counter(class_x[:, i])
            self.classified_feature_prob[c] = feature_counter
        print(self.classified_feature_prob)

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
                    mu, sigma = norm.fit(np.asarray(class_x[:, i], dtype=np.float))
                    feature_counter[i] = {'mu': mu, 'sigma': sigma}
                else:  # 离散特征
                    feature_counter[i] = cc.Counter(class_x[:, i])
            self.classified_feature_prob[c] = feature_counter

    def naive_bayes_classifier(self, x_test):
        '''
        预测测试样本所属类别的概率
        :return:测试类别集
        '''
        x_test = np.asarray(x_test)
        if self.is_binned:
            return self._binned_predict_proba(x_test)
        else:
            return self._gaussian_predict_proba(x_test)

    def _binned_predict_proba(self, x_test):
        '''
        连续特征变量进行分箱离散化，预测
        :param x_test: 测试样本集
        :return:
        '''
        if self.is_feature_all_R:
            x_test = self.dbw.transform(x_test)
        elif self.feature_R_idx is not None:
            x_test = self._data_bin_wrapper(x_test)
        # 存储测试样本所属各个类别的概率
        y_test_hat = np.zeros((x_test.shape[0], self.n_class))
        for i in range(x_test.shape[0]):
            test_sample = x_test[i]  # 当前测试样本
            y_hat = []  # 当前测试样本所属各个类别的概率
            for c in self.class_values:
                prob_ln = np.log(self.prior_prob[c])  # 当前类别的先验概率，取对数
                # 当前类别下不同特征变量不同取值的频次， 构成字典
                feature_frequency = self.classified_feature_prob[c]
                for j in range(x_test.shape[1]):  # 针对每个特征变量
                    value = test_sample[j]  # 当前测试样本的当前特征取值
                    cur_feature_freq = feature_frequency[j]
                    # 按照拉普拉斯修正方法计算
                    prob_ln += np.log(cur_feature_freq.get(value, 0) + 1) / \
                               (self.class_values_num[c] + self.feature_values_num[j])
                y_hat.append(prob_ln)  # 输入第 c类别的概率
            y_test_hat[i, :] = self.softmax_func(np.asarray(y_hat))  # 适合多分类，且归一化
        return y_test_hat

    @staticmethod
    def softmax_func(x):
        '''
        softmax 函数 ，为避免上溢或下溢，对参数x做限制
        :param x: batch_size * n_classes
        :return: 1*n_classes
        '''
        exps = np.exp(x - np.max(x))  # 避免溢出，每个数减去其最大值
        exp_sum = np.sum(exps)
        return exps / exp_sum

    def _gaussian_predict_proba(self, x_test):
        '''
        连续特征变量不进行分箱，直接按高斯分布估计
        :param x_test:  # 测试样本集
        :return:
        '''
        '''
              连续特征变量进行分箱离散化，预测
              :param x_test: 测试样本集
              :return:
              '''
        # if self.is_feature_all_R:
        #     x_test = self.dbw.transform(x_test)
        # elif self.feature_R_idx is not None:
        #     x_test = self._data_bin_wrapper(x_test)
        # 存储测试样本所属各个类别的概率
        y_test_pred = np.zeros((x_test.shape[0], self.n_class))
        for i in range(x_test.shape[0]):
            test_sample = x_test[i]  # 当前测试样本
            y_hat = []  # 当前测试样本所属各个类别的概率
            for c in self.class_values:
                prob_ln = np.log(self.prior_prob[c])  # 当前类别的先验概率，取对数
                # 当前类别下不同特征变量不同取值的频次， 构成字典
                feature_frequency = self.classified_feature_prob[c]
                for j in range(x_test.shape[1]):  # 针对每个特征变量
                    value = test_sample[j]  # 当前样本的当前特征的取值
                    if self.feature_R_idx is not None and (j in self.feature_R_idx):  # 连续特征
                        # 取极大似然估计的均值和方差
                        mu, sigma = feature_frequency[j].values()
                        prob_ln += np.log(norm.pdf(value, mu, sigma) + 1e-7)
                    else:
                        cur_feature_freq = feature_frequency[j]
                        # 按照拉普拉斯修正方法计算
                        prob_ln += np.log((cur_feature_freq.get(value, 0) + 1) / \
                                          (self.class_values_num[c]) + self.feature_values_num[j])

                y_hat.append(prob_ln)  # 输入第 c类别的概率
            y_test_pred[i, :] = self.softmax_func(np.asarray(y_hat))  # 适合多分类，且归一化
        return y_test_pred

    def predict(self, x_test):
        '''
        预测测试样本的所有类别
        :param x_test:
        :return:
        '''
        return np.argmax(self.predict_proba(x_test), axis=1).reshape(-1)

    def predict_proba(self, x_test):
        '''
        预测测试样本所属类别概率
        :param X: 测试样本
        :return:
        '''
        if self.is_binned:
            return self._binned_predict_proba(x_test)
        else:
            return self._gaussian_predict_proba(x_test)
