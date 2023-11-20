import numpy as np
import scipy as sp


class LDAMultic_DimReduction:
    '''
    线性判别分析多分类降维算法
    '''

    def __init__(self, n_components=2):
        self.n_components = n_components  # 降维后的维度数目
        self.class_nums = None  # 类别数
        self.Sw, self.Sb = None, None  # 类间和类内散度矩阵
        self.eig_values = None  # Sb~(-1)*Sb 的特征值
        self.W = None  # 投影矩阵

    def fit(self, x_samples, y_target):
        '''
        LDA 多分类降维核心算法
        :param x_samples:
        :param y_target:
        :return:
        '''
        x_samples, y_target = np.array(x_samples), np.asarray(y_target)
        labels_values = np.unique(y_target)
        self.class_nums = len(labels_values)
        n_features = x_samples.shape[1]
        self.Sw = np.zeros((n_features, n_features))
        mu_t = np.mean(x_samples, axis=0)
        for i in range(self.class_nums):
            class_xi = x_samples[y_target == labels_values[i]]
            mu = np.mean(class_xi, axis=0)
            self.Sw += (class_xi - mu).T.dot(class_xi - mu)
        self.Sb = (x_samples - mu_t).T.dot(x_samples - mu_t) - self.Sw
        # 计算 sb*w=lambda *Sw*w的广义特征值求解
        self.eig_values, eig_vec = sp.linalg.eig(self.Sb, self.Sw)
        idx = np.argsort(self.eig_values)[::-1]  # 逆序索引，从大到小排序
        self.eig_values = self.eig_values[idx]
        vec_sort = eig_vec[:, idx]  # 对特征向量按照特征值从大到小排序的索引排序
        self.W = vec_sort[:, :self.n_components]  # 取前d个特征向量构成投影矩阵w
        return self.W

    def transform(self, x_samples):
        '''
        根据投影矩阵对样本数据进行降维
        :param x_samples:
        :return:
        '''
        if self.W is not None:
            return x_samples.dot(self.W)
        else:
            raise ValueError('请先进行fit,后transform....')

    def fit_transform(self, x_samples, y_targets):
        '''
        获得投影矩阵并降维
        :param x_samples:
        :param y_targets:
        :return:
        '''
        self.fit(x_samples, y_targets)
        return x_samples.dot(self.W)

    def variance_explained(self):
        '''
        各降维成分占总解释方差比
        :return:
        '''
        idx = np.argwhere(np.imag(self.eig_values) != 0)
        if len(idx) == 0:
            self.eig_values = np.real(self.eig_values)
        ratio = self.eig_values / np.sum(self.eig_values)
        return ratio[:self.n_components]


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

irir = load_iris()
X, y = irir.data, irir.target
lda_dr = LDAMultic_DimReduction(n_components=2)
X_new = lda_dr.fit_transform(X, y)
print(lda_dr.variance_explained())
