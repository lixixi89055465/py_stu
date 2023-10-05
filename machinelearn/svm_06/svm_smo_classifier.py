# -*- coding: utf-8 -*-
# @Time    : 2023/10/4 13:44
# @Author  : nanji
# @Site    : 
# @File    : svm_smo_classifier.py
# @Software: PyCharm 
# @Comment :
from machinelearn.svm_06 import kernel_func
import numpy as np
import copy
import random


class SVMClassifier:
    '''
    支持向量机二分类算法：硬间隔，软间隔，核函数，可做线性可分、非线性可分。SMO算法
    1.训练样本的目标值仅限定编码为{0,1},SVM在fit时，把0类别重编码为-1
    '''

    def __init__(self, C=1.0, gamma=1.0, degree=3, coef0=1.0, alpha_tol=1e-3, \
                 kernel='', kkt_tol=1e-3, max_epochs=100):
        self.C = C  # 软间隔硬间隔的参数，硬间隔：适当增大C的值，软间隔：减少C值，允许部分样本不满足的约束条件
        self.gamma = gamma  # 径向基函数/高斯和函数超参数
        self.degree = degree  # 多项式函数的阶次
        self.coef0 = coef0  # 多项式和函数的常数项
        self.kernel = kernel  # 选择的和函数
        if self.kernel is None or self.kernel.lower() == 'linear':
            self.kernel_func = kernel_func.linear()
        elif self.kernel.lower() == 'poly':
            self.kernel_func = kernel_func.poly(degree, coef0)  # self.
            pass
        elif self.kernel.lower() == 'rbf':
            pass
        else:
            print("仅限linear、poly或rbf,值为None 默认为linear线性和函数 ...")
            self.kernel_func = kernel_func.linear()
        self.kkt_tol = kkt_tol  # 在精度内检查
        self.max_epochs = max_epochs
        self.alpha = None  # 松弛变量
        self.alpha_tol = alpha_tol  # 全局支持向量的容忍度
        self.E = None  # 误差
        self.w, self.b = None, None  # SVM的模型系数
        self.support_vectors = []  # 记录支持向量的索引
        self.support_vectors_alpha = []  # 支持向量所对应的松弛变量
        self.support_vectors_x, self.support_vectors_y = [], []  # 支持向量所对应的样本和目标
        self.cross_entropy_loss = []  # 优化过程中的交叉熵损失

    def init_params(self, x_train, y_train):
        '''
        初始化必要参数
        :param x_train: 训练集
        :param y_train: 编码后的目标集
        :return:
        '''
        n_samples, n_features = x_train.shape
        self.w, self.b = np.zeros(n_features), 0.0  # 模型系数初始化为0值
        self.alpha = np.zeros(n_samples)  # 松弛变量
        self.E = self.decision_func(x_train) - y_train  # 初始化误差，所有样本类别取反

    def decision_func(self, x):
        '''
        SVM模型的预测计算，
        :param x_train:
        :return:
        '''
        if len(self.support_vectors) == 0:  # 当前没有支持向量
            if x.ndim <= 1:  # 标量或单个样本
                return 0
            else:
                return np.zeros(x.shape[0])  # np.zeros((x.shape[:-1]))
        else:
            if x.ndim <= 1:
                wt_x = 0.0  # 模型w^T*x+b的第一项求和
            else:
                wt_x = np.zeros(x.shape[0])
            for i in range(len(self.support_vectors)):
                # 公式 : w^T * x = sum( alpha_i * y_i * k(xi , x))
                wt_x += self.support_vectors_alpha[i] * self.support_vectors_y[i] * \
                        self.kernel_func(x, self.support_vectors_x[i])
            return wt_x + self.b

    def _meet_kkt(self, x_i, y_i, alpha_i):
        '''
        判断当前需要优化的alpha是否满足 kkt条件
        :param x_i: 已选择的样本
        :param y_i: 已选择的目标
        :param alpha_i: 需要优化的alpha
        :return:  bool:满足 True,不满足 False
        '''
        if alpha_i < self.C:
            return y_i * self.decision_func(x_i) >= 1 - self.kkt_tol
        else:
            return y_i * self.decision_func(x_i) <= 1 - self.kkt_tol

    def _select_best_j(self, best_i):
        '''
        基于已经选择的第一个alpha_i,寻找使得误差|E_i-E_j|最大的best_j
        :param best_i: 已选择的第一个alpha_i索引
        :return:
        '''
        valid_j_list = [j for j in range(len(self.alpha)) if self.alpha[j] > 0 and best_i != j]
        if len(valid_j_list) > 0:
            idx = np.argmax(np.abs(self.E[best_i] - self.E[valid_j_list]))  # 查找绝对误差最大的索引
            best_j = valid_j_list[int(idx)]  # 最佳的j
        else:
            idx = list(range(len(self.alpha)))  # 所有样本索引
            seq = idx[:best_i] + idx[best_i + 1:]  # 排除best_id
            best_j = random.choice(seq)  # 随机选择
        return best_j

    def _clip_alpha_j(self, y_i, y_j, alpha_j_unc, alpha_i_old, alpha_j_old):
        '''
        修剪第2个更新的alpha值
        :param y_i: 当前选择的第1个y目标值
        :param y_j: 当前选择的第2个y目标值
        :param alpha_j_unc: 当前未修建的第2个alpha值
        :param alpha_i_old: 当前选择的第1个未更新的alpha值
        :param alpha_j_old: 当前选额的第2个未更新的alpha值
        :return:
        '''
        if y_i == y_j:
            inf = max(0, alpha_i_old + alpha_j_old - self.C)  # L
            sup = max(self.C, alpha_i_old + alpha_j_old)  # H
        else:
            inf = max(0, alpha_j_old - alpha_i_old)  # L
            sup = min(self.C, self.C + alpha_j_old - alpha_i_old)  # H
        if alpha_j_unc < inf:
            alpha_j_new = inf
        elif alpha_j_unc > sup:
            alpha_j_new = sup
        else:
            alpha_j_new = alpha_j_unc
        return alpha_j_new

    def _update_error(self, x_train, y_train, y):
        '''
        更新误差，计算交叉熵损失
        :param x_train: 训练样本集
        :param y_train:  目标集
        :param y: 编码后的目标集
        :return:
        '''
        y_predict = self.decision_func(x_train)  # 当前优化过程中的模型预测值
        self.E = y_predict - y  # 误差
        loss = -(y_train.T.dot(np.log(self.sigmoid(y_predict))) + \
                 (1 - y_train).T.dot(np.log(1 - self.sigmoid(y_predict))))
        self.cross_entropy_loss.append(loss / len(y))  # 平均交叉熵损失

    def fit(self, x_train, y_train):
        '''
        SVM 的训练：
        1.按照启发式方法选择一对需要优化的alpha
        2.对参数alpha,b,w,E 等仅限更新，修建
        :param x_train:
        :param y_train:
        :return:
        '''
        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        class_values = np.sort(np.unique(y_train))  # 类别的不同取值
        if class_values[0] != 0 or class_values[1] != 1:
            raise ValueError("仅限二分类，类别编码为{0,1}...")
        y = copy.deepcopy(y_train)
        y[y == 0] = -1  # svm类别仅限定为(-1,1)
        self.init_params(x_train, y)  # 参数的初始化
        n_sample = x_train.shape[0]  # 样本量
        for epoch in range(self.max_epochs):
            if_all_match_kkt_condition = True  # 表示所有样本都满足kkt条件
            # 1.选择一对需要优化的alpha_i 和 alpha_j
            for i in range(n_sample):  # 外层循环
                x_i, y_i = x_train[i], y[i]  # 当前选择的第1个需要优化的alpha所对应的索引
                alpha_i_old, err_i_old = self.alpha[i], self.E[i]  # 取当前为更新的alpha和误差
                if not self._meet_kkt(x_i, y_i, alpha_i_old):  # 不满足 kkt条件
                    j = self._select_best_j(i)  # 基于alpha_i 选额alpha_j
                    alpha_j_old, err_j_old = self.alpha[j], self.E[j]  # 当前第2个
                    x_j, y_j = x_train[j], y[j]  # 第2个需要优化的alpha和误差
                    # 2.基于已经选择的alpha_i和alpha_j，对alpha,b,E和w进行更新
                    # 首先获取未修建的第2个需要更新的alpha值，并对其进行修剪
                    k_ii, k_jj, k_ij = self.kernel_func(x_i, x_i), \
                                       self.kernel_func(x_j, x_j), \
                                       self.kernel_func(x_i, x_j)

                    eta = k_ij + k_jj - 2 * k_ij
                    if eta < 1e-3:  # 避免分母国小，表似乎选择更新的两个样本比较接进行
                        continue
                    alpha_j_unc = alpha_j_old + y_j * (err_i_old - err_j_old) / eta  # 未修建的alpha_j
                    # 修建alpha_j使得满足 0<alpha_j<C
                    alpha_j_new = self._clip_alpha_j(y_i, y_j, alpha_j_unc, alpha_i_old, alpha_j_old)
                    # 3.通过修建后的alpha_j_new 更新alpha_i
                    alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)
                    self.alpha[i], self.alpha[j] = alpha_i_new, alpha_j_new  # 更新回存
                    # 4.更新模型系数w和b.
                    alpha_i_delta, alpha_j_delta = alpha_i_new - alpha_i_old, alpha_j_new - alpha_j_old
                    # w的更新仅与已更新的一对alpha有关
                    self.w = self.w + alpha_i_delta * y_i * x_i + alpha_j_delta * y_j * x_j
                    b_i_delta = -self.E[i] - y_i * k_ii * alpha_i_delta - y_j * k_ij * alpha_j_delta
                    b_j_delta = -self.E[j] - y_i * k_ij * alpha_i_delta - y_j * k_jj * alpha_j_delta
                    if 0 < alpha_i_new < self.C:
                        self.b += b_i_delta
                    elif 0 < alpha_j_new < self.C:
                        self.b += b_j_delta
                    else:
                        self.b += (b_i_delta + b_j_delta) / 2
                    # 5.更新误差E
                    self.support_vectors = np.where(self.alpha > self.alpha_tol)[0]
                    self.support_vectors_x = x_train[self.support_vectors, :]
                    self.support_vectors_y = y[self.support_vectors]
                    self.support_vectors_alpha = self.alpha[self.support_vectors]

                    # 6.更新误差E,计算损失
                    self._update_error(x_train, y_train, y)

            if if_all_match_kkt_condition:  # 没有需要优化的alpha,则提取停机
                break

    def get_params(self):
        '''
        模型SVM的模型系数 ，
        :return:
        '''
        return self.w, self.b

    def predict_proba(self, x_test):
        '''
        预测测试样本所属类别的概率
        :param x_test:测试样本集
        :return:
        '''
        x_test = np.asarray(x_test)
        y_test_hat = np.zeros((x_test.shape[0], 2))  # 存储每个样本的预测概率
        y_test_hat[:, 1] = self.sigmoid(self.decision_func(x_test))
        y_test_hat[:, 0] = 1.0 - y_test_hat[:, 1]
        return y_test_hat

    def predict(self, x_test):
        '''
        预测测试样本所属类别
        :param x_test: 测试样本集
        :return:
        '''
        return np.argmax(self.predict_proba(x_test), axis=1)

    @staticmethod
    def sigmoid(x):
        '''
        sigmoid函数，为避免上溢或下溢，对参数x做限制
        :param x: 可能是标量数据，也可能是数组
        :return:
        '''
        x = np.asarray(x)  # 为避免标量值的布尔索引出错，转化为数组
        x[x > 30.0] = 30.0  # 避免下溢，0.00000000000065
        x[x < -50.0] = -50.0  # 避免上溢 ,1.9287e-22
        return 1 / (1 + np.exp(-x))
