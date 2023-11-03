# -*- coding: utf-8 -*-
# @Time    : 2023/11/2 21:44
# @Author  : nanji
# @Site    : 
# @File    : GMM_Cluster.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import pandas as pd


class GaussianMixtureCluster:
    '''
    高斯混合聚类算法
    '''

    def __init__(self, n_components=3, tol=1e-5, max_epochs=100):
        '''
        使用EM算法训练GMM
        :param n_components: 高斯混合模型数量
        :param tol: 停止迭代的精度要求
        :param max_epochs: 最大迭代次数
        '''
        self.n_compoents = n_components
        self.tol = tol
        self.max_epochs = max_epochs
        self.params = []  # 高斯混合模型参数 ：alpha,mu,sigma

    def gaussian_nd(self, X, mu, sigma):
        '''
        高维高斯分布
        :param X: 样本数据 m*n
        :param mu: 1*n
        :param sigma: n*n
        :return:
        '''
        left_part = 1.0 / (np.power(2 * np.pi, X.shape[1] / 2) * np.sqrt(np.linalg.det(sigma)))
        right_part = np.exp(np.sum(-.5 * (X - mu).dot(np.linalg.inv(sigma)) * (X - mu), axis=1))
        return left_part * right_part

    def fit_gmm(self, X):
        '''
        高斯混合聚类核心算法，实质就是根据后验概率gamma不断更新alpha、均值、协方差
        :param X:
        :return:
        '''
        alpha, mu, sigma = 1.0 / self.n_compoents, np.mean(X, axis=0), np.cov(X.T)
        max_value, min_value = X.max(), X.min()
        for i in range(self.n_compoents):
            self.params.append([alpha, mu + np.random.random() * (max_value + min_value) / 2, sigma])
        # 迭代过程，不断更新alpha,均值，协方差
        gamma, current_log_loss = self.cal_gamma(X)  # 计算后验概率，隐变量Zj,PM
        for _ in range(self.max_epochs):
            pass


    def cal_gamma(self, X):
        '''
        计算后验概率
        :param X:
        :return:
        '''
        gamma = np.array([self.gaussian_nd(X, mu, sigma) * alpha \
                          for alpha, mu, sigma in self.params])
        log_loss = np.sum(np.log(np.sum(gamma, axis=1))) / X.shape[0]
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
        return gamma, log_loss


if __name__ == '__main__':
    X = pd.read_csv('../data/watermelon4.0.csv').values
    gmm = GaussianMixtureCluster()
    gmm.fit_gmm(X)
