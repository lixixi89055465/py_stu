import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    '''
    逻辑回归：采用梯度下降算法+正则化，交义熵损失函数，实现二分类
    样都
    '''

    def __init__(self, fit_intercept=True, normalized=True, alpha=0.05, eps=1e-10, \
                 max_epochs=300, batch_size=20, l1_ratio=None, l2_ratio=None, en_rou=None):
        '''
        param eps:提前停止训练的精度要求，按照两次训练损失的绝对值差小于eps,停止训练
        :param fit_intercept:是否训练偏置项
        :param normalized:是否标准化
        :param alpha:学习
        :param max_epochs:最大的达f代次数
        :param batch_size:批量大小：如果为1，则为随机梯度下降算法,若为总的训练样本数，\
            则为批量梯度下降法，否则是小批量
        :param l1_ratio:LASS0回归惩罚项系数
        :param l2_ratio:岭回归惩罚项系数
        :param en_rou:弹性网络权衡L1L2的系数
        '''
        self.fit_intercept = fit_intercept  # 是否训练偏置项
        self.normalized = normalized  # 是否付样本进行标准化
        self.alpha = alpha  # 学习奉
        self.eps = eps  # 提前停止训练
        if l1_ratio:
            if l1_ratio < 0:
                raise ValueError("惩罚项系数不能为负数...")
        self.l1_ratio = l1_ratio  # LASS0回归惩罚项系数
        if l2_ratio:
            if l2_ratio < 0:
                raise ValueError("惩罚项系数不能为负数..")
        self.l2_ratio = l2_ratio  # 岭回归惩罚项系数
        if en_rou:
            if en_rou > 1 or en_rou < 0:
                raise ValueError("弹性网络权衡系数rou范围在[0,1]")
        self.en_rou = en_rou  # 弹性网客权衡L1L2的系数
        self.max_epoch = max_epochs
        self.batch_size = batch_size
        self.theta = None  # 模型的系数

        if self.normalized:
            # 如果需要标准化，则计算样本特征的的值和标准方差，以便对薄试样本标准化，模型系数的还原
            self.feature_mean, self.feature_std = None, None
            self.n_samples, self.n_features = 0, 0  # 样本量和特征屈性数目
            self.train_loss, self.test_1oss = [], []  # 存储训练过程中的训练损失和测试损失

    @staticmethod
    def sign_func(weight):

        '''
        符号函数，针对l1正则化
        :param weight: 模型系数
        :return:
        '''
        sign_values = np.zeros_like(weight)
        # np.argwhere(weight>0)返回值是索引下标
        sign_values[np.where(weight > 0)] = 1
        sign_values[np.where(weight < 0)] = -1
        return sign_values

    def init_params(self, n_features):
        '''

        :param n_features:
        :return:
        '''
        self.theta = np.random.randn(n_features, 1) * 0.1

    @staticmethod
    def cal_cross_entropy(y_test, y_prob):
        '''
        计算交叉熵损失
        :param y_test:
        :param y_prob:
        :return:
        '''
        loss = -(y_test.T.dot(np.log(y_prob)) + (1 - y_test).T.dot(np.log(1 - y_prob)))
        return loss

    def fit(self, x_train, y_train, x_test=None, y_test=None):
        '''
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        '''
        train_samples = np.c_[x_train, y_train]  # 组合训练集和目标集，以顺便随机打乱样本顺序
        for epoch in range(self.max_epoch):
            self.alpha *= 0.95
            np.random.shuffle(train_samples)  # 打乱样本顺序，模拟随机数
            batch_nums = train_samples.shape[0] // self.batch_size  # 批次
            for idx in range(batch_nums):
                # 按照小批量大小，选取数据
                batch_xy = train_samples[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_x, batch_y = batch_xy[:, :-1], batch_xy[:, -1:]  # 选取样本和目标值
                # 计算权重的更新增量，包含偏置项
                y_prob_batch = self.sigmoid(batch_x.dot(self.theta))  # 小批量的预测值

    def _fit_gradient_desc(self, x_train, y_train, x_test, y_test):
        '''
              :param x_train:
              :param y_train:
              :param x_test:
              :param y_test:
              :return:
              '''
        train_samples = np.c_[x_train, y_train]  # 组合训练集和目标集，以顺便随机打乱样本顺序
        for epoch in range(self.max_epoch):
            self.alpha *= 0.95
            np.random.shuffle(train_samples)  # 打乱样本顺序，模拟随机数
            batch_nums = train_samples.shape[0] // self.batch_size  # 批次
            for idx in range(batch_nums):
                # 按照小批量大小，选取数据
                batch_xy = train_samples[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_x, batch_y = batch_xy[:, :-1], batch_xy[:, -1:]  # 选取样本和目标值
                # 计算权重的更新增量，包含偏置项
                y_prob_batch = self.sigmoid(batch_x.dot(self.theta))  # 小批量的预测值
                delta = ((y_prob_batch - batch_y).T.dot(batch_x) / self.batch_size).T
                # 计算并添加正则化的部分，不包含偏置项
                dw_reg = np.zeros(shape=(x_train.shape[1] - 1, 1))
                if self.l1_ratio and self.l2_ratio is None:
                    # lasso
                    dw_reg = self.l1_ratio * self.sign_func(self.theta[:-1])
                if self.l2_ratio and self.l1_ratio is None:
                    # lasso
                    dw_reg = self.l2_ratio * self.sign_func(self.theta[:-1])

                if self.l1_ratio and self.l2_ratio and self.en_rou:
                    # 弹性网络
                    dw_reg = self.l1_ratio * self.sign_func(self.theta[:-1]) * self.en_rou + \
                             self.l1_ratio * self.sign_func(self.theta[:-1]) * (1 - self.en_rou)
                delta[:-1] += dw_reg / self.batch_size  # 添加了正则化
                self.theta = self.theta - delta  # 更新模型系数
            # 计算训练过程中的交叉熵损失值
            y_train_prob = self.sigmoid(x_train.dot(self.theta))
            train_cost = self.cal_cross_entropy(y_train, y_train_prob)
            self.test_loss.append(train_cost / x_train.shape[0])
            if x_test is not None and y_test is not None:
                y_test_prob = self.sigmoid(x_test.dot(self.theta))  #
                test_loss = self.cal_cross_entropy(y_test, y_test_prob)
                self.test_loss.append(test_loss / x_test.shape[0])  # 交叉熵损失
            # 两次交叉熵损失均值的差异小于给定的精度，提前停止训练
            if epoch > 10 and (np.abs(self.train_loss[-1] - self.train_loss[-2])) <= self.eps:
                break

    def sigmoid(self, z):
        x = np.asarray(z)  # 为避免标量值的布尔索引出错，转化为数组
        x[x > 30.0] = 30.0  # 避免上溢
        x[x < -30.0] = -30.0  # 避免下溢
        return 1.0 / (1 + np.exp(-z))

    def get_params(self):
        return self.theta

    def predict_prob(self, x_test):
        '''
        预测测试样本的概率，第1列为y=0的概率，第2列是y=1的概率
        :param x_test:  测试样本，n*k
        :return:
        '''
        y_prob = np.zeros(shape=(x_test.shape[0], 2))
        if self.normalized:
            x_test = (x_test - self.feature_mean) / self.feature_std
        if self.fit_intercept:
            x_test = np.c_[x_test, np.ones(shape=x_test.shape[0])]
        y_prob[:, 1] = self.sigmoid(x_test.dot(self.theta)).reshape(-1)
        y_prob[:, 0] = 1 - y_prob[:, 1]
        return y_prob

    def predict(self, x, p=0.5):
        '''
        预测样本类别，默认为大于0.5为1，小于0.5为0
        :param x:
        :param p:
        :return:
        '''
        y_prob = self.predict_prob(x)
        # 布尔值转化为整数，true对应1，false 对应0
        return (y_prob[:, 1] > p).astype(int)

    def plt_loss_curve(self, lab=None, is_show=True):
        if is_show:
            plt.figure(figsize=(7, 5))
        plt.plot(self.cross_entropy_loss, 'k-', lw=1)
        plt.xlabel('Training Epochs ', fontdict={'fontsize': 12})
        plt.ylabel('The Mean of Cross Entropy Loss ', fontdict={'fontsize': 12})
        plt.title('The SVM Loss Curve of Cross Entropy')
        plt.grid(ls=':')
        if is_show:
            plt.show()

