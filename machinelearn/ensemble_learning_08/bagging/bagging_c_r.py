# -*- coding: utf-8 -*-
# @Time    : 2023/10/23 18:47
# @Author  : nanji
# @Site    : 
# @File    : bagging.py
# @Software: PyCharm 
# @Comment :
import numpy as np
from sklearn.metrics import accuracy_score
from machinelearn.decision_tree_04.decision_tree_R \
    import DecisionTreeRegression  # cart
from machinelearn.decision_tree_04.decision_tree_C \
    import DecisionTreeClassifier  # cart
import copy


class BaggingClassifierRegression:
    '''
    1.Bagging 的基本流程：采样出T 个含有m个训练样本的采样集，但然后基于每个采样集训练出一个基学习器，再集成。
    2.预测输出进行结合：Bagging 通常对分类任务采用简单投票法，对回归任务使用简单平均法 。
    3.把回归任务与分类任务集成到一个算法中，有参数task来控制，包外估计OOB控制
    '''

    def __init__(self, base_estimator=None, n_estimators=10, \
                 task='C', OOB=False, feature_importance=None):
        '''
        :param base_estimator:  基学习器
        :param n_estimcators:  基学习器的个数 T
        :param learning_rate: 学习率，降低后续训练的基学习器的权重，避免过拟合
        '''
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        if task.lower() not in ['c', 'r']:
            raise ValueError("Bagging任务仅限分类（C/c),回归(R/r)")
        self.task = task
        # 如果不提供学习起，则默认按照深度为2的决策树作为集分类器
        if self.base_estimator is None:
            if self.task.lower() == 'c':
                self.base_estimator = DecisionTreeRegression()
            elif self.task.lower() == 'r':
                self.base_estimator = DecisionTreeClassifier()
        if type(base_estimator) != list:
            # 同质（同种类型）的分类器
            self.base_estimator = [copy.deepcopy(self.base_estimator) \
                                   for _ in range(self.n_estimators)]
        else:
            # 异质（不同种类型）的分类器
            self.n_estimators = len(self.base_estimator)
        self.OOB = OOB  # 是否进行包外估计
        self.oob_indices = []  # 保存每次又放回采样未被使用的样本索引
        self.y_oob_hat = None  # 包括估计样本能预测值（回归）或预测类别概率（分类）
        self.oob_score = None  # 包括估计的评分，分类和回归
        self.feature_importance = feature_importance
        self.feature_importance_scores = None  # 特征变量的重要性评分
        self.feature_sampling_indices=[]# 针对每个基学习器，存储特征变量抽样索引


    def fit(self, x_train, y_train):
        '''
        Bgging 算法（包括分类和回归）的训练
        :param x_train: 训练集
        :param y_train: 目标集
        :return:
        '''
        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        n_sample, n_features = x_train.shape  # 样本量
        for estimator in self.base_estimator:
            # 1.又放回的随机重采样训练集
            indices = np.random.choice(n_sample, n_sample, replace=True)  # 采样样本呢缩影
            indices = np.unique(indices)
            x_bootstrap, y_bootstrap = x_train[indices], y_train[indices]
            # 2.对特征属性变量进行抽样
            fb_num = int(self.feature_sampling_rate * n_features)  # 抽样特征数
            feature_idx = np.random.choice(n_features, fb_num, replace=False)  # 不放回


            # 3.基于采样数据，训练基学习器
            estimator.fit(x_bootstrap, y_bootstrap)
            # 存储每个基学习器未使用的样本索引
            n_indices = set(np.arange(n_sample)).difference(set(indices))
            self.oob_indices.append(list(n_indices))  # 未参与训练的样本索引
        # 3.包外估计
        if self.OOB:
            if self.task.lower() == 'c':
                self._oob_score_classifier(x_train, y_train)
            else:
                self._oob_score_regressor(x_train, y_train)

    def _oob_score_classifier(self, x_train, y_train):
        '''
        分类任务的包外估计
        :param x_train:
        :param y_train:
        :return:
        '''
        self.y_oob_hat, y_true = [], []
        for i in range(x_train.shape[0]):  # 针对每个训练氧泵
            y_hat_i = []  # 当前样本再每个基学习器下的预测概率，个数未必等于 self.n_estimators
            for idx in range(self.n_estimators):  # 针对每个基学习器
                if i in self.oob_indices[idx]:  # 如果该样本属于外包估计
                    y_hat = self.base_estimator[idx].predict_proba(x_train[i, np.newaxis])
                    y_hat_i.append(y_hat[0])
            if y_hat_i:  # 非空，计算各基学习器预测类别概率的均值
                self.y_oob_hat.append(np.mean(np.c_[y_hat_i], axis=0))
                y_true.append(y_train[i])  # 存储对应的真值
        self.y_oob_hat = np.asarray(self.y_oob_hat)
        self.oob_score = accuracy_score(y_true, np.argmax(self.y_oob_hat, axis=1))

    def _oob_score_regressor(self, x_train, y_train):
        '''
        回归任务的包外估计
        :param x_train:
        :param y_train:
        :return:
        '''
        self.y_oob_hat, y_true = [], []
        for i in range(x_train.shape[0]):  # 针对每个训练氧泵
            y_hat_i = []  # 当前样本再每个基学习器下的预测概率，个数未必等于 self.n_estimators
            for idx in range(self.n_estimators):  # 针对每个基学习器
                if i in self.oob_indices[idx]:  # 如果该样本属于外包估计
                    y_hat = self.base_estimator[idx].predict(x_train[i, np.newaxis])
                    y_hat_i.append(y_hat[0])
            if y_hat_i:  # 非空，计算各基学习器预测类别概率的均值
                self.y_oob_hat.append(np.mean(y_hat_i))
                y_true.append(y_train[i])  # 存储对应的真值
        self.y_oob_hat = np.asarray(self.y_oob_hat)
        self.oob_score = accuracy_score(y_true, np.argmax(self.y_oob_hat, axis=1))

    def predict_proba(self, x_test):
        '''
        分类任务中测试样本所属类别的概率预测
        :param x_test:
        :return:
        '''
        if self.task.lower() != 'c':
            raise ValueError("predict_proba（）仅适用于分类任务!")
        x_test = np.asarray(x_test)
        y_test_hat = []  # 用于存储测试样本所属类别概率
        for estimator in self.base_estimator:
            y_test_hat.append(estimator.predict_proba(x_test))
        return np.mean(y_test_hat, axis=0)

    def predict(self, x_test):
        '''
        分类让你无，预测测试样本所属类别，类被概率大者索引为所属类别
        回归任务：预测测试样本，对每个测试样本的预测值简单平均
        :param x_test:
        :return:
        '''
        if self.task.lower() == 'c':
            return np.argmax(self.predict_proba(x_test), axis=1)
        elif self.task.lower() == 'r':
            y_hat = []  # 预测值
            for estimator in self.base_estimator:
                y_hat.append(estimator.predict(x_test))
            return np.mean(y_hat, axis=0)
