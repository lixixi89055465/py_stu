# -*- coding: utf-8 -*-
# @Time    : 2023/9/23 上午10:29
# @Author  : nanji
# @Site    : 
# @File    : lda_multi_dim_reduction.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import scipy

class LDAMulti_DimReduction:
    '''
    线性判断分析多分类降维
    '''

    def __init__(self, n_components=2):
        self.n_components = n_components  # 降维后确定的维度
        self.Sw, self.Sb = None, None
        self.eig_values = None  # 广义特征值
        self.W = None  # 投影矩阵

    def fit(self, x_samples, y_target):
        '''
        线性判别分析多分类降维核心算法，计算投影矩阵W
        :param x_train:
        :param y_train:
        :return:
        '''
        x_train, y_train = np.asarray(x_samples), np.asarray(y_target)
        class_values = np.sort(np.unique(y_train))  # 不同类别的取值
        n_samples, n_features = x_train.shape  # 样本量和特征变量数
        self.Sw = np.zeros((n_features, n_features))
        for i in range(len(class_values)):
            class_x = x_samples[y_target == class_values[i]]
            mu = np.mean(class_x, axis=0)
            self.Sw += (class_x - mu).T.dot(class_x - mu)
        mu_t = np.mean(x_samples, axis=0)
        self.Sb = (x_samples - mu_t).T.dot(x_samples - mu_t) - self.Sw
        self.eig_values,eig_vec=scipy.linalg.eig(self.Sb,self.Sw)
        idx=np.argsort(self.eig_values[::-1])# 从大到小
        self.eig_values=self.eig_values[idx]
        vec_sort=eig_vec[:,idx]
        self.W=vec_sort[:,:self.n_components]
        return self.W

    def transform(self,x_sample):
        '''
        根据投影矩阵计算降维后的新样本数据
        :param X_sample:
        :return:
        '''
        if self.W is not None:
            return x_sample.dot(self.W)
        else:
            raise ValueError("请先进行fit,构造投影矩阵，然后降维。。。")

    def fit_transform(self,x_sample,y_target):
        '''
        计算投影矩阵并及降维
        :param x_sample:
        :param y_target:
        :return:
        '''
        self.fit(x_sample,y_target)
        return x_sample.dot(self.W)
    def variance_explained(self):
        '''
        解释方差比
        :return:
        '''
        idx=np.argwhere(np.imag(self.eig_values)!=0)
        if len(idx)==0:
            self.eig_values=np.real(self.eig_values)

        ratio=self.eig_values/np.sum(self.eig_values)
        return ratio[:self.n_components]



