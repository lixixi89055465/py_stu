# -*- coding: utf-8 -*-
# @Time    : 2023/10/27 13:54
# @Author  : nanji
# @Site    : 
# @File    : RegularizationLinearRegression.py
# @Software: PyCharm 
# @Comment :
import numpy as np


class RegularizationLinearRegression:
    '''
    线性回归+正则化，包含闭式解与梯度下降算法，如果正则化系数L1_ratio或L2_ratio仅传一个
    则相应地为L1或L2，如果正则化系数L1_ratio和L2_ratio都不为空，则采用弹性网络
    '''

    def __init__(self, fit_intercept=True, solver='grad', normalized=True,  #
                 max_epochs=300, alpha=1e-2, batch_size=20, L1_ratio=None,  #
                 L2_ratio=None, en_rou=None):
        '''
        :param fit_intercept:
        :param solver:
        :param nomalized:
        :param max_epochs:
        :param alpha:
        :param batch_size:
        :param L1_ratio:
        :param L2_ratio:
        :param en_rou:
        '''
        self.theta = None  # 训练权重系数
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.normalized = normalized
        if normalized:
            self.feature_mean, self.feature_std = None, None
        self.max_epochs, self.alpha = max_epochs, alpha
        self.batch_size = batch_size
        self.L1_ratio = L1_ratio
        self.L2_ratio = L2_ratio
        self.en_rou = en_rou
        self.train_loss, self.test_loss = [], []  # 均方误差
        self.mse = np.infty
        self.r2, self.r2_adj = 0., 0.
        self.n_sample, self.n_feature = 0, 0

    def fit(self, x_train, y_train, x_test=None, y_test=None):
        '''

        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        '''
        self.n_sample, self.n_feature = x_train.shape
        if self.normalized:
            self.feature_mean = np.mean(x_train, axis=0)
            self.feature_std = np.std(x_train, axis=0) + 1e-8
            x_train = (x_train - self.feature_mean) / self.feature_std
        if self.fit_intercept:
            x_train = np.c_[x_train, np.ones_like[y_train]]
        self.init_params(self.n_feature + 1)
        # 训练模型
        if self.solver == 'form':
            self._fit_closed_form_solution(x_train, y_train, x_test, y_test)
        elif self.solver == 'grad':
            self._fit_sgd(x_train, y_train, x_test, y_test)

    def init_params(self, n_features):
        return np.random.randn(n_features, 1) * 0.1

    def _fit_closed_form_solution(self, x_train, y_train, x_test, y_test):
        '''
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        '''
        if self.L1_ratio is None and self.L2_ratio is None:
            self.theta = np.linalg.pinv(x_train).dot(y_train)
        elif self.L1_ratio is None and self.L2_ratio is not None:
            self.theta = np.linalg.inv(x_train.T.dot(x_train) + self.alpha * np.eye(self.n_feature)).dot(x_train.T).dot(
                y_train)
            self.theta = self.theta.reshape(-1, 1)
        elif self.L1_ratio is not None and self.L2_ratio is None:
            pass
        else:
            self._fit_sgd(x_train, y_train, x_test, y_test)

    def _fit_sgd(self, x_train, y_train, x_test, y_test):
        '''
        梯度下降求解 :根据batch_size 选择随机、批量或小批量算法
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        '''
        train_sample = np.c_[x_train, y_train]  # 组合训练集和目标集，以便随机打乱样本
        # 按 batch_size 更新theta,三种梯度法取决于 batch_size 的大小
        for _ in range(self.max_epochs):
            self.alpha *= 0.95
            np.random.shuffle(train_sample)
            for idx in range(train_sample.shape[0] // self.batch_size):
                batch_xy = train_sample[self.batch_size * \
                                        idx:self.batch_size * (idx + 1)]
                batch_x, batch_y = batch_xy[:, :-1], batch_xy[:, -1:]
                delta = batch_x.T.dot(batch_x.dot(self.theta) - batch_y) / \
                        self.batch_size
                d_reg = np.zeros(shape=(x_train.shape[1] - 1, 1))
                if self.L1_ratio is None and self.L2_ratio is not None:
                    d_reg += self.L1_ratio * np.sign(self.theta[:-1])
                if self.L2_ratio is None and self.L1_ratio is not None:
                    d_reg += self.L2_ratio * self.theta[:-1]
                if self.en_rou and self.L1_ratio and self.L2_ratio:
                    d_reg += self.en_rou * self.L1_ratio * np.sign(self.theta[:-1]) / \
                             self.batch_size  #
                    d_reg += (1 - self.en_rou) * self.L2_ratio * self.theta[:-1] \
                             * 2 / self.batch_size
                d_reg = np.concatenate([d_reg, np.asarray([[0]])], axis=0)
                delta += d_reg
                self.theta = self.theta - self.alpha * self.theta
            y_pred = x_train.dot(self.theta)
            self.train_loss.append(np.mean((y_pred.reshape(-1) - \
                                            y_test.reshape(-1)) ** 2))
            if x_test is not None and y_test is not None:
                y_test_pred = self.predict(x_test)
                mse = np.mean((y_test_pred.reshape(-1) - y_test.reshape(-1)) ** 2)
                self.test_loss.append(mse)

    def predict(self, x_test):
        '''
        测试数据预测
        :param x_test:
        :return:
        '''
        if self.normalized:
            x_test = (x_test - self.feature_mean) / self.feature_std
        if self.fit_intercept:
            x_test = np.c_[x_test, np.ones(shape=(x_test.shape[0]))]
        return x_test.dot(self.theta).reshape(-1, 1)

    def cal_mse_r2(self, y_test, y_pred):
        '''

        :param y_test:
        :param y_pred:
        :return:
        '''
        y_pred, y_test = y_pred.reshape(-1), y_test.reshape(-1)
        self.mse = ((y_pred - y_test) ** 2).mean()
        self.r2 = 1 - ((y_test - y_pred) ** 2).sum() / \
                  ((y_test - y_test.mean()) ** 2).sum()
        self.r2_adj = 1 - (1 - self.r2) * (len(y_test) - 1) / \
                      (len(y_test) - self.n_feature - 1)
        return self.mse, self.r2, self.r2_adj
