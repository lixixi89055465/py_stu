# -*- coding: utf-8 -*-
# @Time    : 2023/10/22 8:47
# @Author  : nanji
# @Site    : 
# @File    : gradientboosting_c.py
# @Software: PyCharm 
# @Comment :  https://www.bilibili.com/video/BV1Nb4y1s7nV/?p=78&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=50305204d8a1be81f31d861b12d4d5cf

import numpy as np
from machinelearn.decision_tree_04.decision_tree_R \
    import DecisionTreeRegression  # cart
import copy


class Gradientboosting_c:
    '''
    提升树回归算法：采用平方误差损失
    '''

    def __init__(self, base_estimator=None, n_estimators=10, learning_rate=1.0,
                 ):
        '''
        :param base_estimator:  基学习器
        :param n_estimcators:  基学习器的个数 T
        :param learning_rate: 学习率，降低后续训练的基学习器的权重，避免过拟合
        :param loss: 损失函数： linear,square,exp
        :param comb_strategy:weight_median、weight_mean
        '''
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        # 如果不提供学习起，则默认按照深度为2的决策树作为集分类器
        if self.base_estimator is None:
            self.base_estimator = DecisionTreeRegression(max_depth=2)
        if type(base_estimator) != list:
            # 同质（同种类型）的分类器
            self.base_estimator = [copy.deepcopy(self.base_estimator) \
                                   for _ in range(self.n_estimators)]
        else:
            # 异质（不同种类型）的分类器
            self.n_estimators = len(self.base_estimator)
        self.base_estimators = []  # 扩展到class_num组分类器

    def fit(self, x_train, y_train):
        '''
        梯度提升分类算法的训练，共训练M*K个基学习器
        :param x_train: 训练集，二维数组；m*k
        :param y_train: 目标集
        :return:
        '''
        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        class_num = len(np.unique(y_train))  # 类别数
        y_encoded = self.one_hot_encoding(y_train)  # one-hot 编码
        # 深拷贝class_num组分类器，每组（每个类别）n_estimators个基学习器
        # 假设是三分类：[[0,1,2,...,9],[10],[10]]
        self.base_estimators = [copy.deepcopy(self.base_estimator) \
                                for _ in range(class_num)]
        # 初始化第一轮基学习器，针对每个类别，分别训练一个基学习器
        y_pred_score_ = []  # 用于存储每个类别的预测值
        for c_idx in range(class_num):
            self.base_estimators[c_idx][0].fit(x_train, y_encoded[:, c_idx])
            y_pred_score_.append(self.base_estimators[c_idx][0].predict(x_train))
        y_pred_score_ = np.c_[y_pred_score_].T  # 把每个类别的预测值构成一列，(n_samples*class_num)
        grad_y = y_encoded - self.softmax_func(y_pred_score_)
        # 训练后续基学习器 ，共M-1轮，每轮针对每个类别，分别训练一个基学习器
        for idx in range(1, self.n_estimators):
            y_hat_values = []  # 用于存储每个类别的预测值
            for c_idx in range(class_num):
                self.base_estimators[c_idx][idx].fit(x_train, grad_y[:, c_idx])
                y_hat_values.append(self.base_estimators[c_idx][idx].predict(x_train))
            y_pred_score_ += np.c_[y_hat_values].T * self.learning_rate  # 把每个类别的预测值构成一列，（n_samples*class_num)
            grad_y = y_encoded - self.softmax_func(y_pred_score_)

    def predict_proba(self, x_test):
        '''
        预测测试样本所属类别的概率
        :param x_test: 测试样本集
        :return:
        '''
        x_test = np.asarray(x_test)
        y_hat_scores = []
        for c_idx in range(len(self.base_estimators)):
            # 获取当前类别的M个基学习器
            estimator = self.base_estimators[c_idx]
            y_hat_scores.append(
                np.sum(  # 每个类别共M个基学习器
                    [estimator[0].predict(x_test)] + \
                    [self.learning_rate * estimator[i].predict(x_test) \
                     for i in range(1, self.n_estimators - 1)] + \
                    [estimator[-1].predict(x_test)], axis=0)
            )
            # y_hat_scores的维度(3*30)
        return self.softmax_func(np.c_[y_hat_scores].T)

    def predict(self, x_test):
        '''
        预测测试样本所属类别，概率大的idx标记为类别
        :param x_test:  测试样本集
        :return:
        '''
        prob = self.predict_proba(x_test)
        print(prob.shape)
        return np.argmax(prob, axis=1)

    @staticmethod
    def one_hot_encoding(target):
        '''
        类别编码
        :param target:
        :return:
        '''
        class_labels = np.unique(target)  # 类别编码，去重
        target_y = np.zeros((len(target), len(class_labels)), dtype=np.int)
        for i, label in enumerate(target):
            target_y[i, label] = 1  # 对应类别所在的列为1
        return target_y

    @staticmethod
    def softmax_func(x):
        '''
        softmax 函数 ，为避免上溢或下溢，对参数x做限制
        :param x: batch_size * n_classes
        :return: 1*n_classes
        '''
        exps = np.exp(x - np.max(x))  # 避免溢出，每个数减去其最大值
        exp_sum = np.sum(exps, axis=1, keepdims=True)
        return exps / exp_sum
