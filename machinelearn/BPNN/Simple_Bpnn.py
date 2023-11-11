# -*- coding: utf-8 -*-
# @Time    : 2023/11/10 22:38
# @Author  : nanji
# @Site    : 
# @File    : Simple_Bpnn.py
# @Software: PyCharm 
# @Comment :
import numpy as np

import machinelearn.BPNN.activity_utils as af
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class SimpleNeuralNetwork:
    '''
    单层神经网络：无隐层
    '''

    def __init__(self, eta=1e-2, precision=None, gradient_method='SGD', \
                 optimizer_method=None, activity_fun='sigmoid', epochs=1000):
        '''
        :param eta: 学习率
        :param precision: 停机精度
        :param gradient_method: 梯度方法：SGD、BGD、MBGD　...
        :param optimizer_method: 优化函数：动量法、adagrad、adam...
        :param activity_fun: 激活函数，默认sigmoid
        :param epochs: 最大训练次数
        '''
        self.eta = eta
        self.precision = precision
        self.gradient_method = gradient_method
        self.optimizer_method = optimizer_method
        #
        self.activity_fun = af.activity_functions(activity_fun)
        self.epochs = epochs
        self.W = None  # 神经网络的权重
        self.loss_values = []  # 每次训练的损失

    def forward(self, x):
        '''
        正向传播
        :param x:
        :return:
        '''
        val = np.dot(self.W.T, x)

    def backward(self, y_true, y_hat):
        '''
        反向传播
        :param y_true:
        :param y_hat:
        :return:
        '''
        error = y_true - y_hat
        delta = self.activity_fun[1](y_hat) * error  # 广义增量规则
        return delta

    def fit_net(self, x_train, y_train):
        '''
        神经网络模型训练，只有一个输入节点
        :param x_train:
        :param y_train:
        :return:
        '''
        n_samples, n_feature = x_train.shape
        np.random.seed(0)
        self.W = np.random.randn(n_feature) / 100  # 初始网络权重
        # 神经树络的训练过程
        for epoch in range(self.epochs):
            if self.gradient_method == "SGD":  # 菜随机梯度下降算法
                x_y = np.c_[x_train, y_train]  # 合并样本
                np.random.shuffle(x_y)

                for i in range(n_samples):
                    x, y_true = x_y[i, :-1], x_y[i, -1]
                    # 正向传播
                    val = np.dot(self.W, x)  # 输入层与权重的点击运算
                    y_hat_sgd = self.activity_fun[0](val)  # 激活函数后的输出值
                    # 反向传播
                    delta = self.backward(y_true, y_hat_sgd)
                    dw = self.eta * delta * x  # 权重更新增量
                    self.W = self.W + dw  # 更新权重
            elif self.gradient_method == "BGD":  # 批显梯度下降法
                y_hat_sgd = self.activity_fun[0](np.dot(self.W, x_train.T))
                delta = self.backward(y_train, y_hat_sgd)
                dw = self.eta * delta.reshape(-1, 1) * x_train
                self.W = self.W + np.mean(dw, axis=0)
                # print("W:", self.W)

            elif self.gradient_method == "MBGD":  # 小批量梯度下法
                pass
            else:
                raise AttributeError("梯度下降方法选择有误：SGD、BGD、MBGD.,.")
            # 平方和损失函数
            y_pred = self.activity_fun[0](np.dot(self.W, x_train.T))
            loss_mean = np.mean((y_pred - y_train) ** 2)
            # 停机规则，两次训练误差损失差小于给定的精度，即停止训练
            # print('loss_mean:', loss_mean)
            self.loss_values.append(loss_mean)
            if self.precision and len(self.loss_values) > 2 and \
                    np.abs(self.loss_values[-1] - self.loss_values[-2]) < self.precision:
                break
        # plt.plot(self.loss_values)
        # plt.savefig('Simple_Bpnn.py.png')
        # # plt.savefig
        # plt.show()

    def plt_loss_curve(self, legend=None):
        '''
        绘制神经网络的损失下降曲线
        :param legend:是否添加图例
        :return:
        '''
        plt.plot(self.loss_values, lw=1.5, label=legend)
        plt.xlabel('Epochs', fontdict={'fontsize': 12})
        plt.ylabel('Loss', fontdict={'fontsize': 12})
        plt.title('Simple Neural Network Loss Curve', fontdict={'fontsize': 14})
        if legend:
            plt.legend()
        plt.grid(ls=':')

    def predict_prob(self, x_test):
        '''
        预测样本属于某个类别的概率
        :param x_test: 测试样本
        :return:
        '''
        y_hat_prob = []
        for i in range(x_test.shape[0]):
            y_prob = self.activity_fun[0](np.dot(self.W, x_test[i]))
            y_hat_prob.append([1 - y_prob, y_prob])
        return np.array(y_hat_prob)

    def predict(self, x_test):
        '''

        :return:
        '''
        y_hat_prob = self.predict_prob(x_test)
        return np.argmax(y_hat_prob, axis=1)


if __name__ == '__main__':
    # X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    # y = np.array([0, 0, 1, 1])
    # plt.figure(figsize=(8, 6))
    # # snn = SimpleNeuralNetwork(epochs=10000,gradient_method="SGD", eta=0.05)
    # snn = SimpleNeuralNetwork(epochs=10000, gradient_method="SGD", \
    #                           activity_fun='tanh', eta=0.05)
    # snn.fit_net(X, y)
    # y_hat = snn.predict_prob(X)
    # print('随机梯度下降法训练的最终结果')
    # print(y_hat)
    # snn.plt_loss_curve(legend='SGD')
    #
    # # snn = SimpleNeuralNetwork(eta=0.05,precision=1e-5, epochs=40000, gradient_method='BGD')
    # snn = SimpleNeuralNetwork(eta=0.05, epochs=10000, \
    #                           gradient_method='BGD', activity_fun='tanh')
    # snn.fit_net(X, y)
    # y_hat = snn.predict_prob(X)
    # print('批量梯度下降法训练的最终结果')
    # snn.plt_loss_curve(legend='BGD')
    # print(y_hat)
    # # plt.savefig('Simple_Bpnn.py.png')
    # plt.savefig('Simple_Bpnn.py.tanh.png')
    # plt.show()
    # iris = load_iris()
    # X, y = iris.data[:100], iris.target[:100]

    # X = StandardScaler().fit_transform(X)
    # y = LabelEncoder().fit_transform(y)
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, \
    #                                                     random_state=0, stratify=y)
    # snn = SimpleNeuralNetwork(eta=0.05, epochs=1000, activity_fun='tanh', \
    #                           gradient_method='SGD')
    # snn.fit_net(x_train, y_train)
    # y_pred = snn.predict(x_test)
    # print(classification_report(y_test, y_pred))
    # snn.plt_loss_curve()
    # plt.savefig("3.png")
    # plt.show()
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    X = StandardScaler().fit_transform(X)
    y = LabelEncoder().fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, \
                                                        random_state=0, stratify=y)
    plt.figure(figsize=(8, 6))
    snn = SimpleNeuralNetwork(epochs=1000, gradient_method="SGD", \
                              activity_fun='tanh', eta=0.05)
    snn.fit_net(X, y)
    y_hat = snn.predict_prob(X)
    print('随机梯度下降法训练的最终结果')
    print(y_hat)
    snn.plt_loss_curve(legend='SGD')

    # snn = SimpleNeuralNetwork(eta=0.05,precision=1e-5, epochs=40000, gradient_method='BGD')
    snn = SimpleNeuralNetwork(eta=0.05, epochs=1000, \
                              gradient_method='BGD', activity_fun='tanh')
    snn.fit_net(X, y)
    y_hat = snn.predict_prob(X)
    print('批量梯度下降法训练的最终结果')
    snn.plt_loss_curve(legend='BGD')
    print(y_hat)
    # plt.savefig('Simple_Bpnn.py.png')
    plt.savefig('4.png')
    plt.show()
