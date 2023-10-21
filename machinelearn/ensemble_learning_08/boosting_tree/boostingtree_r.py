# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/10/21 10:41
# @Author  : nanji
# @File    : boostingtree_r.py
# @Description : https://www.bilibili.com/video/BV1Nb4y1s7nV/?p=74&spm_id_from=pageDriver&vd_source=50305204d8a1be81f31d861b12d4d5cf

import numpy as np
from machinelearn.decision_tree_04.decision_tree_R \
    import DecisionTreeRegression  # cart
import copy


class BoostTreeRegression:
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

    def fit(self, x_train, y_train):
        '''
        训练AdaBoost每个基分类器 ， 计算权重分布，每个基学习器的误差率和权重系数alpha
        :param x_train: 训练集，二维数组；m*k
        :param y_train: 目标集
        :return:
        '''
        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        self.base_estimator[0].fit(x_train, y_train)
        y_hat = self.base_estimator[0].predict(x_train)
        y_residual = y_train - y_hat  # 残差，MSE的负梯度
        # 2.从第2课树开始，每一次拟合上一轮的残差
        for idx in range(1, self.n_estimators):
            self.base_estimator[idx].fit(x_train, y_residual)  # 拟合残差
            # 累加第m-1棵树开始，每一次拟合上一轮的残差
            y_hat += self.base_estimator[idx].predict(x_train) * self.learning_rate
            y_residual = y_train - y_hat  # 当前模型的残差

    def _cal_loss(self, y_true, y_hat):
        '''
        根据损失函数计算相对误差
        @param y_true:  真值
        @param y_hat: 预测值
        @return:
        '''
        errors = np.abs(y_true - y_hat)  # 绝对值误差
        if self.loss.lower() == 'linear':  # 线性
            return errors / np.max(errors)
        elif self.loss.lower() == 'square':  # 平方
            return errors ** 2 / np.max(errors) ** 2
        elif self.loss.lower() == 'exp':  # 指数
            return 1 - np.exp(-errors / np.max(errors))
        else:
            raise ValueError("仅支持 linear、square和exp ...")

    def predict_proba(self, x_test):
        '''
        预测测试样本所属类别的概率，软投票
        :param x_test: 测试样本集
        :return:
        '''
        x_test = np.asarray(x_test)
        # 按照加法模型，现行组合基学习器
        # 每个测试样本，每个基学习器预测概率(10,[(0.68,0.32),(0.55,0.45)]...)
        y_hat_prob = np.sum([self.base_estimator[i].predict_proba(x_test) * \
                             self.estimator_weights[i] for i in \
                             range(self.n_estimators)], axis=0)
        return y_hat_prob / y_hat_prob.sum(axis=1, keepdims=True)

    def predict(self, x_test):
        '''
        AdaBoost 回归算法预测，按照加权中位数以及加权平均两种结合策略
        :param x_test: 测试样本集
        :return:
        '''
        x_test = np.asarray(x_test)
        y_hat_mat = np.sum([self.base_estimator[0].predict(x_test)] + \
                           [np.power(self.learning_rate, i) * self.base_estimator[i].predict(x_test) \
                            for i in range(1, self.n_estimators - 1)] + \
                           [self.base_estimator[-1].predict(x_test)], axis=0)
        return y_hat_mat
