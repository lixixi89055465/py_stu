# -*- coding: utf-8 -*-
# @Time    : 2023/10/24 17:28
# @Author  : nanji
# @Site    : 
# @File    : LinearRegression_GradDesc.py
# @Software: PyCharm 
# @Comment :
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


class LinearRegression_GradDesc:
    def __init__(self, fit_intercept=True, normalize=True, epochs=300, alpha=1e-2, batch_size=20):
        '''
        :param fit_intercept: 是否训练bias
        :param normalize:  是否标准化数据
        :param epochs:  最大迭代次数
        :param alpha: 学习率
        :param batch_size: 小批量样本呢数据，若为1，则为随机梯度，若为训练样本量，则为小批量T恤
        '''
        self.theta = None  # 训练权重系数
        self.fit_intercept = fit_intercept  # 线性模型的常数项，也即偏执bias,模型中的theta0
        self.normalize = normalize  # 是否标准化数据
        if normalize:
            self.feature_mean, self.feature_std = None, None  # 特征的均值，标准方差
        self.epochs, self.alpha = epochs, alpha
        self.train_loss, self.test_loss = [], []
        self.batch_size = batch_size
        self.mse = np.infty  # 测试样本的均方误差
        self.r2, self.r2_adj = 0., 0.
        self.n_samples, self.n_features = 0, 0  # 样本数和特征数

    def init_params(self, n_features):
        '''
        初始化参数
        :param n_features:
        :return:
        '''
        self.theta = np.random.random(size=(n_features, 1)) * 0.1

    def fit(self, x_train, y_train, x_test=None, y_test=None):
        '''
        模型训练，根据是否标准化与是否拟合偏执项分类讨论
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        '''
        if self.normalize:
            self.feature_mean = np.mean(x_train, axis=0)  # 样本均值
            self.feature_std = np.std(x_train, axis=0) + 1e-8  #
            x_train = (x_train - self.feature_mean) / self.feature_std
        self.init_params(x_train.shape[1])
        self._fit_sgd(x_train, y_train, x_test, y_test)  # 梯度训练

    def predict(self, x_test):
        '''

        :param x_test:
        :return:
        '''
        self.n_samples, self.n_features = x_test.shape[0], x_test.shape[1]
        if self.normalize:
            x_test=(x_test-self.feature_mean)/self.feature_std
        if self.fit_intercept:
            x_test=np.c_[x_test,np.ones(self.n_samples)]
        return x_test.dot(self.theta).reshape(-1,1)

    def _fit_sgd(self, x_train, y_train, x_test, y_test):
        '''
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        '''
        train_sample = np.c_[x_train, y_train]
        best_theta, best_mse = None, np.infty
        for i in range(self.epochs):
            self.alpha *= 0.95
            np.random.shuffle(train_sample)
            for idx in range(train_sample // self.batch_size):
                batch_xy = train_sample[self.batch_size * idx:self.batch_size * (idx + 1)]
                batch_x, batch_y = batch_xy[:, :-1], batch_xy[:, -1:]
                delta = batch_x.T.dot(batch_x.dot(self.theta) - batch_y) / self.batch_size
                self.theta = self.theta - self.alpha * delta
            train_mse = ((x_train.dot(self.theta) - y_train) ** 2).mean()
            self.train_loss.append(train_mse)
            if x_test is not None and y_test is not None:
                y_test_pred = self.predict(x_test)
                test_mse = ((x_test.dot(self.theta) - y_test) ** 2).mean()
                if test_mse < best_mse:
                    best_mse = test_mse
                    best_theta = np.copy(self.theta)
                self.test_loss.append(test_mse)
        if best_theta is None:
            self.theta = np.copy(best_theta)
    def cal_mse_r2(self,y_pred,y_test):
        self.mse=((y_pred.shape(-1,1)-y_pred)**2).mean()
        self.r2=1-self.mse.sum()/((y_test.reshape(-1,1)-y_test.mean)**2).sum()
        self.r2_adj=1-(1-self.r2)*(self.n_samples-1)/(self.n_samples-self.n_features-1)
        return self.mse,self.r2,self.r2_adj

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



