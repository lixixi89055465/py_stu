# -*- coding: utf-8 -*-
# @Time    : 2023/10/23 16:42
# @Author  : nanji
# @Site    : 
# @File    : LRCloseFormSol.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import matplotlib.pyplot as plt


class LRCloseFormSol:
    def __init__(self, fit_intercept=True, normalize=True):
        '''
        :param fit_intercept: 是否训练bias
        :param normalize:  是否标准化
        '''
        self.theta = None  # 训练权重系数
        self.fit_intercept = fit_intercept  # 象形模型的常数值，即bias,模型中的theta0
        self.normalize = normalize  # 是否标准化数据
        if normalize:
            self.feature_mean, self.feature_std = None, None
        self.mse = np.infty  # 训练样本呢的均方误差
        self.r2, self.r2_adj = 0.0, 0.0  # 判定系数和修正绑定系数
        self.n_sample, self.n_features = 0, 0  # 样本书和特征数

    def fit(self, x_train, y_train):
        '''
        模型训练，根据是否标准化与是否拟合偏置项进行分类讨论
        :param x_train:
        :param y_train:
        :return:
        '''
        if self.normalize:
            self.feature_mean = np.mean(x_train, axis=0)  # 均值
            self.feature_std = np.std(x_train, axis=0)   # 方差
            x_train = (x_train - self.feature_mean) / self.feature_std  # 标准化
        if self.fit_intercept:
            x_train = np.c_[x_train, np.ones_like(y_train)]  # 添加一列1,即偏置项样本
        # 训练模型
        self._fit_closed_form_solution(x_train, y_train)

    def _fit_closed_form_solution(self, x_train, y_train):
        '''

        :param x_train: n_sample*n_feature
        :param y_train: n_sample*n_feature
        :return:
        '''
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        # self.theta = np.linalg.pinv(x_train).dot(y_train)
        self.theta = np.linalg.pinv(x_train).dot(y_train)

    def get_params(self):
        '''
        返回线性模型训练的系数
        :return:
        '''
        if self.fit_intercept:
            weight, bias = self.theta[:-1], self.theta[-1]
        else:
            weight, bias = self.theta, np.array([0])

        if self.normalize:  # 标准化后的系数
            weight = weight / (self.feature_std.reshape(-1))
            bias = bias - weight.T.dot(self.feature_mean.reshape(-1))
        return np.r_[weight.reshape(-1), bias.reshape(-1)]

    def predict(self, x_test):
        '''
        侧手数据预测，x_test:待预测样本集，不包括偏置项1
        :param x_test:
        :return:
        '''
        try:
            self.n_sample, self.n_features = x_test.shape[0],x_test.shape[1]
        except IndexError:
            self.n_sample, self.n_features = x_test.shape[0], 1
        if self.normalize:
            x_test = (x_test - self.feature_mean) / (self.feature_std+1e-8)
        if self.fit_intercept:
            x_test = np.c_[x_test, np.ones(shape=(x_test.shape[0]))]
        return x_test.dot(self.theta)

    def cal_mse_r2(self, y_test, y_pred):
        '''
        计算均方误差，计算拟合优度的判定系数R方和修正判定系数
        :param y_test: 测试目标真值
        :param y_pred: 模型预测目标真值
        :return:
        '''
        self.mse = ((y_test - y_pred) ** 2).mean()  #
        self.r2 = 1 - ((y_test - y_pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
        self.r2_adj = 1 - (1 - self.r2) * (self.n_sample - 1) / (self.n_sample - self.n_features - 1)
        return self.mse, self.r2, self.r2_adj

    def plt_predict(self, y_pred, y_test, is_show=True, is_sort=True):
        '''
        预测真值与真实值对比图
        :param y_pred:
        :param y_test:
        :param is_show:
        :param is_sort:
        :return:
        '''
        if self.mse is np.infty:
            self.cal_mse_r2(y_test, y_pred)
        if is_show:
            plt.figure(figsize=(7, 5))
        if is_sort:
            idx = np.argsort(y_test)
            plt.plot(y_pred[idx], "r:", lw=1.5, label="Test value val")
            plt.plot(y_test[idx], "k--", lw=1.5, label="Predictive value val")
        else:
            plt.plot(y_test, "ko-", lw=1.5, label="Test value val")
            plt.plot(y_pred, "r*-", lw=1.5, label="Predictive value val")
        plt.xlabel('Test Sample observation serial number', fontdict={"fontsize": 12})
        plt.ylabel('Predictive Sample value', fontdict={"fontsize": 12})
        plt.title("The mse = %.5e,R2=%.5f,R2_adj=%.5f" % \
                  (self.mse, self.r2, self.r2_adj), fontdict={"fontsize": 12})
        plt.legend(frameon=False)
        plt.grid(ls=":")
        if is_show:
            plt.show()
