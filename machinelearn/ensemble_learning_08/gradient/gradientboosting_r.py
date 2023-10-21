# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/10/21 10:41
# @Author  : nanji
# @File    : gradientboosting_r.py
# @Description : https://www.bilibili.com/video/BV1Nb4y1s7nV/?p=76&spm_id_from=pageDriver&vd_source=50305204d8a1be81f31d861b12d4d5cf
import numpy as np
from machinelearn.decision_tree_04.decision_tree_R \
    import DecisionTreeRegression  # cart
import copy


class GradientBoostRegression:
    '''
    梯度提升回归算法：采用误差损失: 五个，以损失函数在当前模型的负梯度近似为残差
    1.假设回归决策树与mse构建的，针对不同的损失函数，计算不同的基尼指数划分标准
    2.预测，集成，也根据不同的损失函数，预测叶子节点的输出

    '''

    def __init__(self, base_estimator=None, n_estimators=10, learning_rate=1.0,
                 loss='ls', huber_threshold=0.1, quantile_threshold=0.5):
        '''
        :param base_estimator:  基学习器
        :param n_estimcators:  基学习器的个数 T
        :param learning_rate: 学习率，降低后续训练的基学习器的权重，避免过拟合
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
        self.loss = loss  # 损失函数的类型
        self.huber_threshold = huber_threshold  # 仅对huber损失有效
        self.quantile_threshold = quantile_threshold  # 仅对分位数损失有效

    def _cal_negative_gradient(self, y_true, y_pred):
        '''
        计算负梯度值
        @param y_true: 真值
        @param y_pred:  预测值
        @return:
        '''
        if self.loss.lower() == 'ls':  # mse
            return y_true - y_pred
        elif self.loss.lower() == 'lae':  # MAE
            return np.sign(y_true - y_pred)
        elif self.loss.lower() == 'huber':  # 平滑平局绝对损失
            return np.where(np.abs(y_true - y_pred) < self.huber_threshold, \
                            y_true - y_pred, \
                            self.huber_threshold * np.sign(y_true - y_pred))
        elif self.loss.lower() == 'quantile':  # 分位数损失
            return np.where(y_true > y_pred, self.quantile_threshold, self.quantile_threshold - 1)
        elif self.loss.lower() == 'logcosh':  # 双曲余弦的对数的负梯度
            return -np.tanh(y_pred - y_true)
        else:
            raise ValueError("仅限于ls、lae、huber、quantile和logcosh，选择有误...")

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
        y_residual = self._cal_negative_gradient(y_train,y_hat)# 负梯度
        # 2.从第2课树开始，每一次拟合上一轮的残差
        for idx in range(1, self.n_estimators):
            self.base_estimator[idx].fit(x_train, y_residual)  # 拟合残差
            # 累加第m-1棵树开始，每一次拟合上一轮的残差
            y_hat += self.base_estimator[idx].predict(x_train) * self.learning_rate
            y_residual = self._cal_negative_gradient(y_train,y_hat)#负梯度

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
        return y_hat_prob

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
