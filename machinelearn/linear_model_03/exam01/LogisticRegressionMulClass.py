import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class LogisticRegressionMulClass:
    '''

    '''

    def __init__(self, fit_intercept=True, normalize=True, l1_ratio=None, l2_ratio=None, max_epochs=300, eta=0.05,
                 batch_size=20, eps=1e-10):
        self.weight = None  # 模型系数
        self.fit_intercept = fit_intercept  # 是否训练偏置项
        self.no1malize = normalize  # 是否标准化
        if normalize:
            self.feature_mean, self.feature_std = None, None
        self.max_epochs = max_epochs  # 最大送f代次数
        self.eta = eta  # 学习
        self.batch_size = batch_size  # 批量大小，如果为1，则为机梯度下降法
        self.l1_ratio, self.l2_ratio = l1_ratio, l2_ratio  # 正则化系数
        self.eps = eps  # 如果两次相邻训练的损失之差小于精度，则提取停止
        self.train_loss, self.test_loss = [], []  # 训练损失和测试损
        self.n_class = None  # 类别数，即n分类

    def init_params(self, n_feature, n_classes):
        '''
        初始化参数，标准正态缝补，且乘以一个较小的数
        :param n_feature:
        :param n_classes:
        :return:
        '''
        self.weight = np.random.randn(n_feature, n_classes) * 0.05

    @staticmethod
    def one_hot_encoding(target):
        '''
        类别one-hot编码
        :param target:
        :return:
        '''
        class_labels = np.unique(target)
        target_y = np.zeros(shape=(len(target), len(class_labels)))
        for i, label in enumerate(target):
            target_y[i, label] = 1
        return target_y

    @staticmethod
    def softmax_func(logits):
        '''

        :param logits:
        :return:
        '''
        exp = np.exp(logits - np.max(logits))  # 避免上溢和下溢
        exps_sum = np.sum(exp)
        return exp / exps_sum

    @staticmethod
    def cal_cross_entropy(y_true, y_prob):
        '''
        :param self:
        :param y_true:
        :param y_prob:
        :return:
        '''
        loss = -np.sum(y_true * np.log(y_prob + 1e-8), axis=1)
        loss -= np.sum((1 - y_true) * np.log(1 - y_prob + 1e-8), axis=1)
        return np.mean(loss)

    @staticmethod
    def sign_func(weight):
        '''

        :param weight:
        :return:
        '''
        sign = np.zeros_like(weight)
        sign = np.where(weight > 0, 1, weight)
        sign = np.where(weight < 0, -1, sign)
        sign = np.where(weight == 0, 0, sign)
        return sign
