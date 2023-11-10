# -*- coding: utf-8 -*-
# @Time    : 2023/10/15 下午3:38
# @Author  : nanji
# @Site    : 
# @File    : adaboost_discrete_c.py
# @Software: PyCharm 
# @Comment :
import numpy as np
from machinelearn.decision_tree_04.decision_tree_C \
    import DecisionTreeClassifier
import copy


class AdaBoostClassifier:
    '''
    adaboost 分类算法：既可以做而分类，也可以做多分类
    1.同质学习起：非列表形式，按同种类型的基学习器构造
    2.异质学习器：列表传递[lg, svm, cart, ...]
    '''

    def __init__(self, base_estimator=None, n_estimators=10, learning_rate=1.0):
        '''

        :param base_estimator: 
        :param n_estimcators: 
        :param learning_rate: 
        '''
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        # 如果不提供学习起，则默认按照深度为2的决策树作为集分类器
        if self.base_estimator is None:
            self.base_estimator = DecisionTreeClassifier(max_depth=2)
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
            y_hat_0 = (self.base_estimator[idx].predict(x_train) == y_train).astype(int)
            # 2.计算分类误差率
            error_rate = sample_weights.dot(1.0 - y_hat_0) / n_samples
            if error_rate > 0.5:
                self.estimator_weights.append(0)  # 当前分类器不起作用
                continue
            # 3.计算基学习器的权重系数，考虑溢出
            alpha_rate = 0.5 * np.log((1 - error_rate) / (error_rate + 1e-8)) + np.log(n_class - 1)
            alpha_rate = min(10.0, alpha_rate)  # 避免权重系数过大
            self.estimator_weights.append(alpha_rate)
            # 4. 更新样本权重，为了适应多分类，yi*GM(xi)计算np.power(-1.0,1-y_hat_0)
            sample_weights *= np.exp(-1.0 * alpha_rate * np.power(-1.0, 1 - y_hat_0))
            sample_weights = sample_weights / np.sum(sample_weights) * n_samples
        # 5.更新estimcator的权重系数，按照学习率
        for i in range(self.n_estimators):
            self.estimator_weights[i] *= np.power(self.learning_rate, i)

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
        预测测试样本所属类别
        :param x_test:
        :return:
        '''
        return np.argmax(self.predict_proba(x_test), axis=1)
