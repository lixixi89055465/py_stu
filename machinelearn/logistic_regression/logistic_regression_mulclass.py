# -*- coding: utf-8 -*-
# @Time    : 2023/9/17 下午1:45
# @Author  : nanji
# @Site    : 
# @File    : logistic_regression.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class LogisticRegression_MulClass:
    '''
    逻辑回归+正则化 ，梯度下降法+ 正则化，交叉熵损失函数，实现多分类，softmax函数
    '''

    def __init__(self, fit_intercept=True, normalize=True, alpha=5e-2, eps=1e-10,
                 max_epochs=300, batch_size=20, l1_ratio=None, l2_ratio=None, en_rou=None):
        '''
        :param eps: 提前停止训练的精度要求，按照两次训练损失函数的绝对值差小于eps,停止训练。
        :param fit_intercept: 是否训练偏执项想
        :param normalize: 是否标准化
        :param alpha: 学习率
        :param max_epochs: 最大的迭代次数
        :param batch_size: 批量大小，如果为1，则为随机梯度下降算法，若为总的训练样本数，则为批量梯度算法,否则是小批量
        :param l1_ratio: LASSO回归惩罚项系数
        :param l2_ratio: 岭回归惩罚项系数
        :param en_rou: 弹性网络权衡L1和L2的系数
        '''
        self.alpha = alpha
        # self.l1_ratio=l1_ratio #LASsO回归惩罚项系数
        self.eps = eps
        if l1_ratio:
            if l1_ratio < 0:
                raise ValueError('惩罚项系数不能为负数... ')
        self.l1_ratio = l1_ratio
        if l2_ratio:
            if l2_ratio < 0:  # 岭回归惩罚项系数
                raise ValueError('惩罚项系数不能为负数... ')
        self.l2_ratio = l2_ratio  # 岭回归惩罚项系数
        if en_rou:
            if 0 > en_rou or en_rou > 1:
                raise ValueError('弹性网络权衡系数rou范围在[0,1]')
        self.en_rou = en_rou

        self.fit_intercept = fit_intercept  # 是否训练偏执项想
        self.normalize = normalize
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.weight = None  # 模型的系数
        if self.normalize:
            # 如果需要标准化，则计算样本特征的均值和标准方差，以便对测试样本标准化，模型系数的还原
            self.feature_mean, self.feature_std = None, None

        self.n_sample, self.n_classes = 0, 0  # 样本量和类别数目
        self.train_loss, self.test_loss = [], []  # 存储训练过程总的训练损失和测试损失

    def init_params(self, n_features, n_classes):
        '''
        模型参数的初始化

        :param n_features:样本的特征数量
        :param n_class:类别数目
        :return: n_features*n_classes
        '''
        self.weight = np.random.randn(n_features, n_classes) * 0.05

    @staticmethod
    def softmax_func(x):
        '''
        softmax 函数 ，为避免上溢或下溢，对参数x做限制
        :param x: batch_size * n_classes
        :return: 1*n_classes
        '''
        exps = np.exp(x - np.max(x))  # 避免溢出，每个数减去其最大值
        exp_sum = np.sum(exps, axis=1, keepdims=True)
        return exps / exp_sum

    @staticmethod
    def one_hot_encoding(target):
        '''
        类别编码
        :param target:
        :return:
        '''
        class_labels = np.unique(target)  # 类别编码，去重
        target_y = np.zeros((len(target), len(class_labels)), dtype=np.int)
        for i, label in enumerate(target):
            target_y[i, label] = 1  # 对应类别所在的列为1
        return target_y

    @staticmethod
    def sign_func(z_values):
        '''
        符号函数，针对L1正则化
        :param z_values: 模型系数
        :return:
        '''
        z_values[z_values > 0]=1
        z_values[z_values < 0]=-1
        return z_values

    @staticmethod
    def cal_cross_entropy(y_test, y_prob):
        '''
        计算交叉熵损失
        :param y_test: 样本真值，二维数组n*c,c 表示类别数目
        :param y_prob: 模型预测类别概率 n*c
        :return:
        '''
        loss = -np.sum(y_test*np.log(y_prob+1e-8),axis=1)
        loss -= np.sum((1-y_test)*np.log(1-y_prob+1e-8),axis=1)
        return loss.mean()

    def fit(self, x_train, y_train, x_test=None, y_test=None):
        '''
        样本的预处理，模型系数的求解,闭式解公式+梯度方法
        :param x_train: 训练样本 ndarray m*k
        :param y_train: 训练目标值  m*1
        :param X_test: 测试样本 ndarray n*k
        :param y_test: 测试目标值  n*1
        :return:
        '''
        y_train=self.one_hot_encoding(y_train)
        if y_test is not None:
            y_test=self.one_hot_encoding(y_test)
        samples=np.r_[x_train,x_test]# 组合所有样本，计算均值和标准差
        if self.normalize:  # 标准化
            self.feature_mean = np.mean(samples, axis=0)  # 样本的均值
            self.feature_std = np.std(samples, axis=0) + 1e-8  # 样本的标准方差
            x_train = (x_train - self.feature_mean) / self.feature_std  # 标准化
            if x_test is not None and y_test is not None:
                x_test = (x_test - self.feature_mean) / self.feature_std  # 标准化

        if self.fit_intercept:#是否训练bias
            x_train = np.c_[x_train, np.ones((len(x_train),1))]# 在样本后加一列
            if x_test is not None and y_test is not None:
                x_test = np.c_[x_test, np.ones((len(x_test),1))]  # 在样本后加一列1

        self.init_params(n_features=x_train.shape[1],n_classes=y_train.shape[1])  # 模型初始化
        # 训练模型
        self._fit_gradient_descent(x_train, y_train, x_test, y_test)

    def _fit_gradient_descent(self, x_train, y_train, x_test=None, y_test=None):
        '''
        三种梯度下降算法的实现
        :param x_train:
        :param y_train:
        :return:
        '''
        train_sample = np.c_[x_train, y_train]  # 组合训练集和目标集 ，以便随机打乱样本顺序
        # n_features=x_train.shape[1]# 训练样本的特征数目，可能包含偏执项
        for epoch in range(self.max_epochs):
            self.alpha *= 0.95
            np.random.shuffle(train_sample)  # 打乱样本顺序，以便模拟机器
            batch_nums = train_sample.shape[0] // self.batch_size  # 批次
            for idx in range(batch_nums):
                # 按照小批量大小，选取数据
                batch_xy = train_sample[idx * self.batch_size:(idx + 1) * self.batch_size]
                # 选取样本和目标值，注意，目标值不再是一列
                batch_x, batch_y = batch_xy[:, :x_train.shape[1]], batch_xy[:, x_train.shape[1]:]  # 选取样本和目标值
                # delta = batch_x.T.dot((batch_x.dot(self.theta) - batch_y)) / self.batch_size
                # 计算权重的更新增量，包含偏执项
                # delta = batch_x.T.dot(batch_x.dot(self.theta) - batch_y.reshape(-1, 1))
                y_preb_batch = self.softmax_func(batch_x.dot(self.weight))  # 小批量的预测值
                # 1*n <--> n*k = 1*k--> 转置k*1
                dw = ((y_preb_batch - batch_y).T.dot(batch_x)/ self.batch_size).T
                # 计算并添加正则化部分,包含偏置项,不包含偏执项,最后一列是偏执项
                dw_reg = np.zeros(shape=(x_train.shape[1] - 1, self.n_classes))
                if self.l1_ratio and self.l2_ratio is None:
                    # LASSO回归，L1 正则化
                    dw_reg = self.l1_ratio * self.sign_func(self.weight[:-1])
                if self.l2_ratio and self.l1_ratio is None:
                    # Ridege 回归
                    dw_reg = 2 * self.l2_ratio * self.weight[:-1, :]
                if self.en_rou and self.l1_ratio and self.l2_ratio and 0 < self.en_rou < 1:
                    # 弹性网络
                    dw_reg += self.l1_ratio * self.en_rou * self.sign_func(self.weight[:-1, :])
                    dw_reg += 2 * self.l2_ratio * (1 - self.en_rou) * self.weight[:-1, :]
                dw[:-1,:] += dw_reg / self.batch_size  # 添加了正则化
                self.weight = self.weight - self.alpha * dw  # 更新模型系数

            # 计算训练过程中的交叉熵误差损失值
            y_train_prob = self.softmax_func(x_train.dot(self.weight))  # 当前迭代训练的模型预测概率
            train_cost = self.cal_cross_entropy(y_train, y_train_prob)  # 训练集的交叉熵损失
            self.train_loss.append(train_cost)  # 交叉熵损失均值
            if x_test is not None and y_test is not None:
                y_test_preb = self.softmax_func(x_test.dot(self.weight))  # 当前测试样本预测概率
                test_cost = self.cal_cross_entropy(y_test,y_test_preb)
                self.test_loss.append(test_cost )  # 交叉熵损失均值
            if epoch > 10 and (np.abs(self.train_loss[-1] - self.train_loss[-2])) <= self.eps:
                break

    def get_params(self):
        '''
        获取模型的系数
        :return:
        '''
        if self.fit_intercept:
            weight, bias = self.weight[:-1, :], self.weight[-1, :]
        else:
            weight, bias = self.weight, np.array([0])
        if self.normalize:
            weight = weight / self.feature_std.reshape(-1, 1)  # 还原模型系数
            bias = bias - weight.T.dot(self.feature_mean)

        return weight, bias

    def predict_prob(self, x_test):
        '''
        预测测试样本的概率
        :param x_test: 测试样本，ndarray : n*k
        :return:
        '''
        if self.normalize:
            x_test = (x_test - self.feature_mean) / self.feature_std  # 测试样本的标准化
        if self.fit_intercept:
            x_test = np.c_[x_test, np.ones(shape=x_test.shape[0])]
        y_prob= self.softmax_func(x_test.dot(self.weight))
        return y_prob

    def predict(self, x):
        '''
        预测样本类别，默认大于0.5为1，小于0.5为0
        :param x: 预测样本
        :param p: 概率阈值
        :return:
        '''
        y_prob = self.predict_prob(x)
        # 对应每个样本中所有类别的概率，哪个概率大，返回那个类别所在列索引编号，即类别
        return y_prob.argmax(axis=1)

    def plt_cross_entropy_loss(self, lab=None, is_show=True):
        '''
        可视化损失曲线
        :return:
        '''
        # plt.figure(figsize=(7, 5))
        plt.plot(self.train_loss, 'k--', lw=1, label='Train loss')
        if self.test_loss:
            plt.plot(self.test_loss, 'r--', lw=1.2, label='Test loss')
        plt.xlabel('Training Epochs ', fontdict={'fontsize': 12})
        plt.ylabel('The Mean of Cross Entropy Loss ', fontdict={'fontsize': 12})
        plt.title('%s: The loss curve of corss entropy' % lab)
        plt.grid(ls=':')
        plt.legend(frameon=False)
        if is_show:
            plt.show()

    @staticmethod
    def plt_confusion_matrix(confusion_matrix, label_names=None, is_show=True):
        '''
        可视化混淆矩阵

        :param confusion_matrix: 混淆矩阵
        :return:
        '''
        sns.set()
        cm = pd.DataFrame(confusion_matrix, columns=label_names, index=label_names)
        sns.heatmap(cm, annot=True, cbar=False)
        acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
        plt.title("Confusion Matrix and ACC = %.5f" % (acc))
        if is_show:
            plt.show()
