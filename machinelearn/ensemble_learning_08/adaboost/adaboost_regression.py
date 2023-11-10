# -*- coding: utf-8 -*-
# @Time    : 2023/10/15 下午3:38
# @Author  : nanji
# @Site    : 
# @File    : adaboost_discrete_c.py
# @Software: PyCharm 
# @Comment :
import numpy as np
from machinelearn.decision_tree_04.decision_tree_R \
    import DecisionTreeRegression  # cart
import copy


class AdaBoostRegression:
    '''
    adaboost 回归算法：结合（集成）策略，加权中位数、预测值的加权平均
    1.同质学习起：非列表形式，按同种类型的基学习器构造
    2.异质学习器：列表传递[lg, svm, cart, ...]
    3.回归误差率依赖于相对误差：平方误差、线性误差、指数误差

    '''

    def __init__(self, base_estimator=None, n_estimators=10, learning_rate=1.0,
                 loss='square', comb_strategy='weight_median'):
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
        self.loss = loss  # 相对误差的损失函数
        self.comb_strategy = comb_strategy  # 结合策略
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
        self.estimator_weights = []  # 每个基学习器的权重系数

    def fit(self, x_train, y_train):
        '''
        训练AdaBoost每个基分类器 ， 计算权重分布，每个基学习器的误差率和权重系数alpha
        1.基学习器基于权重分布Dt的训练集的训练
        2.计算最大绝对误差、相对误差、回归误差率v
        3.计算当前ht的置信度
        4.更新下一轮的权重分布
        :param x_train: 训练集，二维数组；m*k
        :param y_train: 目标集
        :return:
        '''
        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        n_samples, n_class = x_train.shape[0], len(set(y_train))  # 样本量，类别数
        n_samples = x_train.shape[0]  # 样本量
        sample_weights = np.ones(n_samples)  # 为适应自写的基学习器，设置样本均匀权重为1.0,样本权重
        # 针对每一个学习器，根据带有权重分布的训练集训练基学习器，计算相关参数
        for idx in range(self.n_estimators):
            # 1. 使用只有权重分布Dm的训练数据集学习，并预测
            self.base_estimator[idx].fit(x_train, y_train, sample_weights)
            # 只关心分类错误的，如果分类错误，则为0，正确则为1
            y_hat = self.base_estimator[idx].predict(x_train)  # 当前训练集的预测值
            # 2.计算最大绝对值误差、相对误差、回归误差率
            errors = self._cal_loss(y_train, y_hat)
            error_rate = np.dot(errors, sample_weights / n_samples)  # 回归误差率
            # 3.计算当前ht的置信度，基学习器的权重参数
            alpha_rate = error_rate / (1 - error_rate)
            self.estimator_weights.append(alpha_rate)
            # 4. 更新下一轮的权重分布
            sample_weights *= np.power(alpha_rate, 1 - errors)
            sample_weights = sample_weights / np.sum(sample_weights) * n_samples
        # 5.更新estimcator的权重系数，按照学习率
        for i in range(self.n_estimators):
            self.estimator_weights[i] *= np.power(self.learning_rate, i)

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
        self.estimator_weights = np.asarray(self.estimator_weights)
        x_test = np.asarray(x_test)
        if self.comb_strategy == 'weight_mean':  # 加权平均
            self.estimator_weights /= np.sum(self.estimator_weights)
            # n*T
            y_hat_mat = np.array([self.estimator_weights[i] * self.base_estimator[i].predict(x_test)
                                  for i in range(self.n_estimators)])
            return np.sum(y_hat_mat, axis=0)
        elif self.comb_strategy == 'weight_median':  # 加权中位数
            # Ｔ个基学习器的预测结果构成一个二维数组（３０，１２７）－－＞　（127，30）
            y_hat_mat = np.array([ self.base_estimator[i].predict(x_test)
                                  for i in range(self.n_estimators)]).T
            sorted_idx = np.argsort(y_hat_mat, axis=1)  # 二位数组
            weight_cdf = np.cumsum(self.estimator_weights[sorted_idx], axis=1)
            # 选择最小的t
            median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
            median_idx = np.argmax(median_or_above, axis=1)  # 返回每个样本的t索引值
            median_estimators = sorted_idx[np.arange(x_test.shape[0]), median_idx]
            return y_hat_mat[np.arange(x_test.shape[0]), median_estimators]
