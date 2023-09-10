# -*- coding: utf-8 -*-
# @Time    : 2023/9/9 16:22
# @Author  : nanji
# @Site    :
# @File    : LinearRegression_CFSol.py
# @Software: PyCharm
# @Comment :
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression_GradDesc:
    '''
    线性回归，模型的闭式解
    1.数据的预处理，是否训练偏置项(默认True) ,是否标准化 (默认True)
    2.模型的训练，闭式解公式
    3.模型的预测
    4.均方误差 ,判决系数
    5.模型预测可视化
    '''

    def __init__(self, fit_intercept=True, normalize=True, alpha=1e-2, max_epoch=300, batch_size=20):
        '''

        :param fit_intercept: 是否训练偏执项想
        :param normalize: 是否标准化
        :param alpha: 学习率
        :param max_epoch: 最大的迭代次数
        :param batch_size: 批量大小，如果为1，则为随机梯度下降算法，若为总的训练样本数，则为批量梯度算法,否则是小批量
        '''
        self.alpha = alpha
        self.fit_intercept = fit_intercept  # 是否训练偏执项想
        self.normalize = normalize
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.theta = None  # 模型的系数
        if self.normalize:
            # 如果需要标准化，则计算样本特征的均值和标准方差，以便对测试样本标准化，模型系数的还原
            self.feature_mean, self.feature_std = None, None
        self.mse = np.infty  # 模型预测的均方误差
        self.r2, self.r2_adj = 0.0, 0.0  # 判决系数和修正判决系数
        self.n_sample, self.n_features = 0, 0  # 样本量和特征数目
        self.train_loss, self.test_loss = [], []  # 存储训练过程总的训练损失和测试损失

    def init_params(self, n_features):
        '''
        模型参数的初始化

        :param n_features:样本的特征数量
        :return:
        '''
        self.theta = np.random.randn(n_features, 1) * 0.1

    def fit(self, x_train, y_train, x_test=None, y_test=None):
        '''
        样本的预处理，模型系数的求解,闭式解公式
        :param x_train: 训练样本 ndarray m*k
        :param y_train: 训练目标值  m*1
        :param X_test: 测试样本 ndarray n*k
        :param y_test: 测试目标值  n*1
        :return:
        '''
        if self.normalize:
            self.feature_mean = np.mean(x_train, axis=0)  # 样本的均值
            self.feature_std = np.std(x_train, axis=0) + 1e-8  # 样本的标准方差
            x_train = (x_train - self.feature_mean) / self.feature_std  # 标准化
            if x_test is not None and y_test is not None:
                x_test = (x_test - self.feature_mean) / self.feature_std  # 标准化

        if self.fit_intercept:
            x_train = np.c_[x_train, np.ones_like(y_train)]
            if x_test is not None and y_test is not None:
                x_test = np.c_[x_test, np.ones_like(y_test)]  # 在样本后加一列1

        self.init_params(n_features=x_train.shape[1])  # 模型初始化
        # 训练模型
        # self.__fit_closed_form_solution(x_train=x_train, y_train=y_train)
        self._fit_gradient_descent(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    def _fit_gradient_descent(self, x_train, y_train, x_test=None, y_test=None):
        '''
        三种梯度下降算法的实现
        :param x_train:
        :param y_train:
        :return:
        '''
        train_sample = np.c_[x_train, y_train]  # 组合训练集和目标集 ，以便随机打乱样本顺序

        for i in range(self.max_epoch):
            # self.alpha*=0.99
            np.random.shuffle(train_sample)  # 打乱样本顺序，以便模拟机器
            batch_nums = train_sample.shape[0] // self.batch_size  # 批次
            for idx in range(batch_nums):
                # 按照小批量大小，选取数据
                batch_xy = train_sample[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_x, batch_y = batch_xy[:, :-1], batch_xy[:, -1]  # 选取样本和目标值
                # 计算权重的更新增量
                delta = batch_x.T.dot((batch_x.dot(self.theta) - batch_y)) / self.batch_size
                self.theta = self.theta - self.alpha * delta
            train_mse = ((x_train.dot(self.theta) - y_train.reshape(-1, 1)) ** 2).mean()
            self.train_loss.append(train_mse)  # 每次迭代的训练损失值 （MSE）
            test_mse = ((x_test.dot(self.theta) - y_test.reshape(-1, 1)) ** 2).mean()
            self.test_loss.append(test_mse)

            if x_test is not None and y_test is not None:
                test_mse = ((x_test.dot(self.theta) - y_test.reshape(-1, 1)) ** 2).mean()
                self.test_loss.append(test_mse)  # 每次迭代的训练损失值

        print(self.train_loss)
        print(self.test_loss)

    def plt_predict(self, y_test, y_pred, is_sort=True):
        '''
        预测结果的可视化
        :param y_test: 测试样本真值
        :param y_pred: 测试样本预测值
        :param is_sort: 是否排序 ，然后可视化
        :return:
        '''
        plt.figure(figsize=(7, 5))
        if is_sort:
            idx = np.argsort(y_test)  #
            plt.plot(y_test[idx], 'k-', lw=1.5, label='Test True val')
            plt.plot(y_pred[idx], 'r:', lw=1.8, label='Predict True val')
        else:
            plt.plot(y_test, 'ko-', lw=1.0, label='Test True val')
            plt.plot(y_pred, 'r*-', lw=1.0, label='Predict True val')
        plt.xlabel('Test samples numbers', fontdict={'fontsize': 12})
        plt.ylabel('Predicted samples values', fontdict={'fontsize': 12})
        plt.title('The loss curve of MSE by Gradient descent Method')
        plt.grid(ls=':')
        plt.legend(frameon=False)
        plt.show()

    def plt_loss_curve(self):
        '''
        可视化损失曲线
        :return:
        '''
        plt.figure(figsize=(7, 5))
        plt.plot(self.train_loss, 'k--', lw=1, label='Train loss s')
        if self.test_loss:
            plt.plot(self.test_loss, 'r--', lw=1.2, label='Test loss')
        plt.xlabel('Epochs', fontdict={'fontsize': 12})
        plt.ylabel('loss values ', fontdict={'fontsize': 12})
        plt.title('Gradient Descent Method \n'
                  'Mse = %.5e,R2 = %.5f, R2_adj = %.5f' % (self.mse, self.r2, self.r2_adj))
        plt.grid(ls=':')
        plt.legend(frameon=False)
        plt.show()
