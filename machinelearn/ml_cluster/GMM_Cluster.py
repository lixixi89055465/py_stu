# -*- coding: utf-8 -*-
# @Time    : 2023/11/2 21:44
# @Author  : nanji
# @Site    : 
# @File    : GMM_Cluster.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import seaborn as sns
from sklearn.preprocessing import StandardScaler


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
        gamma, current_log_loss = self.update_gamma(X)  # 计算后验概率，隐变量Zj,PM
        for _ in range(self.max_epochs):
            for k in range(self.n_compoents):
                # 针对每个高斯分布，更新参数
                self.params[k][0] = np.mean(gamma[:, k])  # 更新alpha参数
                # 更新均值
                self.params[k][1] = np.sum(gamma[:, [k]] * X, axis=0) / gamma[:, k].sum()
                # 更新方差
                self.params[k][2] = (gamma[:, [k]] * (X - self.params[k][1])).T. \
                                        dot(X - self.params[k][1]) / gamma[:, k].sum()
            # 计算后验概率
            gamma, new_log_loss = self.update_gamma(X)
            # 终止条件,按照后验概率每次迭代的变化值
            if np.abs(new_log_loss - current_log_loss) > self.tol:
                current_log_loss = new_log_loss
            else:
                break

    def update_gamma(self, X):
        '''
        计算后验概率
        :param X:
        :return:
        '''
        gamma = self.cal_gamma(X)
        log_loss = np.sum(np.log(np.sum(gamma, axis=1))) / X.shape[0]
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
        return gamma, log_loss

    def cal_gamma(self, X):
        '''
        计算高斯分布,p(x|mu,sigma)
        :param X: 样本数据
        :return:
        '''
        return np.array([self.gaussian_nd(X, mu, sigma) * alpha \
                         for alpha, mu, sigma in self.params]).T

    def predict_proba(self, X):
        '''
        预测样本在第1个高斯分布模型上的慨率分布：
        模型参数self.params已经收敛，最终的
        param X:样本数据
        return:m k
        '''
        gamma = self.cal_gamma(X)
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
        return gamma

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def plt_contourf(self):
        plt.figure(figsize=(8, 6))
        x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
        X1, X2 = np.meshgrid(x1, x2)  # 100*100
        X_ = np.c_[X1.reshape(-1, 1), X2.reshape(-1, 1)]
        Z = self.predict_sample_generate_proba(X_)  # 10000*1
        # 绘制等高线
        C = plt.contour(X1, X2, Z.reshape(X1.shape), levels=6, linestyles='--')
        plt.clabel(C, inline=True, fontsize=10)  # 标记等高线值
        labels = self.predict()  # 每个样本所属的高斯分布簇标记
        markers = 'os<>p*h'
        for label in np.unique(labels):
            cluster = X[label == labels]
            plt.scatter(cluster[:, 0], cluster[:, 1], marker=markers[label])
        plt.title('Gaussian Mixture Cluster of EM Algorithm', fontdict={'fontsize': 14})
        plt.xlabel('Feature 1 ', fontdict={'fontsize': 12})
        plt.ylabel('Feature 2 ', fontdict={'fontsize': 12})
        plt.grid(ls=':')
        plt.legend()
        plt.show()

    def predict_sample_generate_proba(self, X):
        '''
        样本的生成概率
        :param X:
        :return:
        '''
        gamma = self.cal_gamma(X)
        return np.sum(gamma, axis=1)


if __name__ == '__main__':
    # X, _ = make_blobs(400, centers=4, cluster_std=0.6, random_state=0)
    # gmm = GaussianMixtureCluster(n_components=4, tol=1e-10)
    # gmm.fit_gmm(X)
    # gmm.plt_contourf()
    # for i in range(len(gmm.params)):
    #     print("alpha:", gmm.params[i][0])
    #     print("mu:", gmm.params[i][1])
    #     print("sigma:", [gmm.params[i][2][0], gmm.params[i][2][1]])
    ##############################################
    # X = pd.read_csv('../data/watermelon4.0.csv').values
    # gmm = GaussianMixtureCluster()
    # gmm.fit_gmm(X)
    # print('0' * 100)
    # gmm.plt_contourf()
    # print('1' * 100)
    # for i in range(len(gmm.params)):
    #     print("alpha:", gmm.params[i][0])
    #     print("mu:", gmm.params[i][1])
    #     print("sigma:", [gmm.params[i][2][0], gmm.params[i][2][1]])
    ##############################################
    X = pd.read_csv('../data/consumption_data.csv')
    X = StandardScaler().fit_transform(X)
    gmm = GaussianMixtureCluster(n_components=3, tol=1e-10)
    gmm.fit_gmm(X)
    labels = gmm.predict(X)
    # gmm.plt_contourf()
    # 可视化核密度估计
    title = ['R index ', 'F index', 'M index']
    plt.figure(figsize=(7, 10))
    for f in range(X.shape[1]):
        plt.subplot(311 + f)
        for c in range(gmm.n_compoents):  # c表示簇索引
            sns.kdeplot(X[labels == c][:, f])
        plt.grid()
        plt.title(title[f])
    plt.show()
    print('1' * 100)
    for i in range(len(gmm.params)):
        print("alpha:", gmm.params[i][0])
        print("mu:", gmm.params[i][1])
        print("sigma:", [gmm.params[i][2][0], gmm.params[i][2][1]])
