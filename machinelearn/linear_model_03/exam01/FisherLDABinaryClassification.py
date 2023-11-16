import numpy as np


class FisherLDABinaryClassification:
    '''
    Fisher判别分析，针对二分类，线性可分数据集
    '''

    def __init__(self):
        self.mu_ = dict()  # 类均值向量
        self.Sw_i = dict()  # 类内离散矩阵
        self.Sw = None  # 总类内离散度矩阵
        self.weight = None  # 投影方向，权重系数
        self.w0 = None  # 阈值

    def fit(self, X_train, y_train):
        '''
        Fisher线性判别分析核心算法，建立模型，获得投影方向w
        :param X_train: 训练样本集
        :param y_train: 训练样本目标集合
        :return:
        '''
        x_train, y_train = np.asarray(X_train), np.asarray(y_train)
        class_label = np.sort(np.unique(y_train))
        if len(class_label) != 2:
            raise ValueError("仅限于二分类且线性可分数据集...")
        # 1.按照类别分类样本，计算各类别均值向量、类内离散度矩阵，总类内离散度矩阵
        sample_size = []  # 存储各类别样本量大小
        self.Sw = np.zeros((x_train.shape[1], x_train.shape[1]))  # 初始化总类内离散度矩阵
        for label in class_label:
            class_samples = x_train[y_train == label]  # 布尔索引，提取类别样本子集
            self.mu_[label] = np.mean(class_samples, axis=0, dtype=np.float)
            sample_size.append(len(class_samples))  # 类内样本量
            self.Sw_i[label] = (class_samples - self.mu_[label]).T \
                .dot(class_samples - self.mu_[label])  # 计算类内离散度矩阵 class_size*class*size
            self.Sw = self.Sw + self.Sw_i[label]  # 总类内离散度矩阵
        # 2.计算投影方向，按奇异值分解的方法以及最佳投影方向公式
        self.Sw = np.array(self.Sw, dtype=np.float)
        u, sigma, v = np.linalg.svd(self.Sw)  # 奇异值分解
        inv_sw = v * np.linalg.inv(np.diag(sigma)) * u.T  # 求Sw的逆矩阵
        self.weight = inv_sw.dot(self.mu_[0] - self.mu_[1])  # 投影方向
        # 3.j计算阈值 w0
        self.w0 = -(sample_size[0] * self.weight.dot(self.mu_[0]) +
                    sample_size[1] * self.weight.dot(self.mu_[1])) / np.sum(sample_size)

    def predict(self, x_test):
        '''
        LDA 判别测试样本类别
        :param x_test: 测试样本
        :return:
        '''
        x_test = np.asarray(x_test)
        # 针对测试样本，计算判别函数值：投影值+ 阈值
        y_pred = self.weight.dot(x_test.T) + self.w0
        y_test_labels = np.zeros(x_test.shape[0])  # 存储测试样本的列别
        y_test_labels[y_pred < 0] = 1  # 小于阈值的为负类
        return y_test_labels


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs, load_iris, load_breast_cancer

center = np.array([[0.3, 1], [1.5, 0.5]])
std = np.array([0.3, 0.3])
X, y = make_blobs(n_samples=2000, n_features=15, centers=center, \
                  cluster_std=std, random_state=1)
x_train, x_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y, random_state=0)
lda = FisherLDABinaryClassification()
lda.fit(X_train=x_train, y_train=y_train)
y_hat = lda.predict(x_test)
print(classification_report(y_test, y_hat))
