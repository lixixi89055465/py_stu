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


class SAMMERClassifier:
    '''
    SAMME.R算法是将SAMME拓展到连续数值型的范畴
    '''

    def __init__(self, base_estimator=None, n_estimators=10):
        '''
        :param base_estimator: 基学习器
        :param n_estimcators: 基学习器 的个数 T
        '''
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
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
        self.n_samples, self.n_class = None, None

    def _target_encoding(self, y_train):
        '''
        对目标值进行编码
        :param y_train:  训练目标集
        :return:
        '''
        self.n_samples, self.n_class = len(y_train), len(set(y_train))
        target = -1.0 / (self.n_class - 1) * \
                 np.ones((self.n_samples, self.n_class))
        for i in range(self.n_samples):
            target[i, y_train[i]] = 1  # 对应该样本的类别所在编码中的列
        return target

    def fit(self, x_train, y_train):
        '''
        训练AdaBoost每个基分类器 ， 计算权重分布，每个基学习器的误差率和权重系数alpha
        :param x_train: 训练集，二维数组；m*k
        :param y_train: 目标集
        :return:
        '''
        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        target = self._target_encoding(y_train)  # 编码
        sample_weights = np.ones(self.n_samples)  # 为适应自写的基学习器，设置样本均匀权重为1.0,样本权重
        # 针对每一个学习器，根据带有权重分布的训练集训练基学习器，计算相关参数
        c = (self.n_class - 1) / self.n_class
        for idx in range(self.n_estimators):
            # 1. 使用只有权重分布Dm的训练数据集学习，并预测
            self.base_estimator[idx].fit(x_train, y_train, sample_weights)
            y_pred = self.base_estimator[idx].predict_proba(x_train)
            np.clip(y_pred,np.finfo(y_pred.dtype).eps,None,out=y_pred)
            # 只关心分类错误的，如果分类错误，则为0，正确则为1
            sample_weights *= np.exp(-c * (target * np.log(y_pred)).sum(axis=1))
            sample_weights /= np.sum(sample_weights) * self.n_samples

    def predict_proba(self, x_test):
        '''
        预测测试样本所属类别的概率，软投票
        :param x_test: 测试样本集
        :return:
        '''
        x_test = np.asarray(x_test)
        C_x = np.zeros((x_test.shape[0], self.n_class))
        for i in range(self.n_estimators):
            y_prob = self.base_estimator[i].predict_proba(x_test)
            np.clip(y_prob, np.finfo(y_prob.dtype).eps, None, out=y_prob)
            y_ln = np.log(y_prob)
            C_x += (self.n_class - 1) * (y_ln - 1.0 / self.n_class * np.sum(y_ln, axis=1, keepdims=True))
        return self.softmax_func(C_x)

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

    def predict(self, x_test):
        '''
        预测测试样本所属类别
        :param x_test:
        :return:
        '''
        x_test = np.array(x_test)
        C_x = np.zeros((x_test.shape[0], self.n_class))
        for i in range(self.n_estimators):
            y_prob = self.base_estimator[i].predict_proba(x_test)
            np.clip(y_prob, np.finfo(y_prob.dtype).eps, None, out=y_prob)
            y_ln = np.log(y_prob)
            C_x += (self.n_class - 1) * (y_ln - np.sum(y_ln, axis=1, keepdims=True) / self.n_class)
        return np.argmax(C_x, axis=1)
