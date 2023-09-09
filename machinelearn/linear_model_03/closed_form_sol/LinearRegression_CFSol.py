# -*- coding: utf-8 -*-
# @Time    : 2023/9/9 16:22
# @Author  : nanji
# @Site    : 
# @File    : LinearRegression_CFSol.py
# @Software: PyCharm 
# @Comment :
import numpy as np


class LinearRegressionClosedFormSol:
    '''
    线性回归，模型的闭式解
    1.数据的预处理，是否训练偏置项(默认True) ,是否标准化 (默认True)
    2.模型的训练，闭式解公式
    3.模型的预测
    4.均方误差 ,判决系数
    5.模型预测可视化
    '''

    def __init__(self, fit_intercept=True, normalized=True):
        self.fit_intercept = fit_intercept
        self.normalized = normalized
        self.theta = None  # 模型的系数
        if self.normalized:
            # 如果需要标准化，则计算样本特征的均值和标准方差，以便对测试样本标准化，模型系数的还原
            self.feature_mean, self.feature_std = None, None
        self.mse = np.inf # 模型预测的均方误差
        self.r2,self.r2_adj=0.0,0.0 # 判决系数和修正判决系数
        self.n_sample,self.n_features=0,0 # 样本量和特征数目

    def fit(self,x_train,y_train):
        '''
        样本的预处理，模型系数的求解,闭式解公式
        :param x_train: 训练样本 ndarray m*k
        :param y_train: 目标值  m*1
        :return:
        '''
        if self.normalized:
            self.feature_mean=np.mean(x_train,axis=0)# 样本的均值
            self.feature_std=np.std(x_train,axis=0)#样本的标准方差
            x_train=(x_train-self.feature_mean)/self.feature_std #标准化
        if self.fit_intercept:
            x_train=np.c_(x_train,np.ones_like(y_train))#在样本后加1列 1
    def __fit_closed_form_solution(self,x_train,y_train):
        '''
          模型系数的求解,闭式解公式
        :param x_train: 数据预处理后的训练样本 ndarray m*k
        :param y_train: 目标值  ndarray , m*1
        :return:
        '''
        self.theta=np.linalg.pinv(x_train,)
        # xtx=np.dot(x_train.T,x_train)+0.0001*np.eye(x_train.shape[1])#防止不可逆
        # self.theta=np.linalg.inv(xtx).dot(x_train.T).dot(x_train)
    def get_params(self):
        '''
        获取模型的系数
        :return:
        '''
        if self.fit_intercept:
            weight,bias=self.theta[:-1],self.theta[-1]
        else:
            weight,bias=self.theta,np.array(0)
        if self.normalized:
            weight=weight/self.feature_sd# 还原模型的系数
            bias=bias-weight.T.dot(self.feature_mean)
        return weight,bias

    def predict(self,x_test):
        '''
        模型的预测
        :param x_test: 测试样本
        :return:
        '''
        if self.normalized:
            x_test=(x_test-np.mean(x_test,axis=0))/np.std(x_test,axis=0)#测试样本的标准化
        if self.fit_intercept:
            x_test=np.c_(x_test,np.ones(x_test.shape[0]))
        return x_test.dot(self.theta)












