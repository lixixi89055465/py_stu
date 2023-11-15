import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from machinelearn.model_evaluation_selection_02.Performance_metrics import ModelPerformanceMetrics


class LogisticRegressionMulClass:
    @staticmethod
    def plt_confusion_matrix(confusion_matrix, label_names=None, is_show=True):
        '''

        :param confusion_matrix:
        :param label_names:
        :param is_show:
        :return:
        '''
        sns.set()
        cm = pd.DataFrame(confusion_matrix, columns=label_names, index=label_names)
        sns.heatmap(cm, annot=True, cbar=False)
        acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
        plt.title('Confusion Matrix and ACC = %.5f ' % (acc))
        if is_show:
            plt.show()

    def __init__(self, fit_intercept=True, normalize=True, l1_ratio=None, \
                 l2_ratio=None, max_epochs=300, eta=0.05,\
                 batch_size=20, eps=1e-10):
        self.weight = None  # 模型系数
        self.fit_intercept = fit_intercept  # 是否训练偏置项
        self.normalize = normalize  # 是否标准化
        if normalize:
            self.feature_mean, self.feature_std = None, None
        self.max_epochs = max_epochs  # 最大送f代次数
        self.eta = eta  # 学习
        self.batch_size = batch_size  # 批量大小，如果为1，则为机梯度下降法
        self.l1_ratio, self.l2_ratio = l1_ratio, l2_ratio  # 正则化系数
        self.eps = eps  # 如果两次相邻训练的损失之差小于精度，则提取停止
        self.train_loss, self.test_loss = [], []  # 训练损失和测试损
        self.n_class = None  # 类别数，即n分类

    def init_params(self, n_feature, n_classes):
        '''
        初始化参数，标准正态缝补，且乘以一个较小的数
        :param n_feature:
        :param n_classes:
        :return:
        '''
        self.weight = np.random.randn(n_feature, n_classes) * 0.05

    @staticmethod
    def one_hot_encoding(target):
        '''
        类别one-hot编码
        :param target:
        :return:
        '''
        class_labels = np.unique(target)
        target_y = np.zeros(shape=(len(target), len(class_labels)))
        for i, label in enumerate(target):
            target_y[i, label] = 1
        return target_y

    @staticmethod
    def softmax_func(logits):
        '''

        :param logits:
        :return:
        '''
        exp = np.exp(logits - np.max(logits))  # 避免上溢和下溢
        exps_sum = np.sum(exp, axis=1, keepdims=True)
        return exp / exps_sum

    @staticmethod
    def cal_cross_entropy(y_true, y_prob):
        '''
        :param self:
        :param y_true:
        :param y_prob:
        :return:
        '''
        loss = -np.sum(y_true * np.log(y_prob + 1e-8), axis=1)
        loss -= np.sum((1 - y_true) * np.log(1 - y_prob + 1e-8), axis=1)
        return np.mean(loss)

    @staticmethod
    def sign_func(weight):
        '''

        :param weight:
        :return:
        '''
        sign = np.where(weight > 0, 1, weight)
        sign = np.where(weight < 0, -1, sign)
        sign = np.where(weight == 0, 0, sign)
        return sign

    def fit(self, x_train, y_train, x_test, y_test):
        '''
        训练模型，并根据是否标准化和是否训练bias进行不同的判断
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        '''
        y_train = self.one_hot_encoding(y_train)
        y_test = self.one_hot_encoding(y_test)
        samples = np.r_[x_train, x_test]  # 组合所有样本，计算均值和方差
        if self.normalize:
            self.feature_mean = np.mean(samples, axis=0)
            self.feature_std = np.std(samples, axis=0) + 1e-8
            x_train = (x_train - self.feature_mean) / self.feature_std
            x_test = (x_test - self.feature_mean) / self.feature_std
        if self.fit_intercept:  # 是否训练bias
            x_train = np.c_[x_train, np.ones(shape=(x_train.shape[0], 1))]  # 拼接一列 1
            x_test = np.c_[x_test, np.ones(shape=(x_test.shape[0], 1))]  # 拼接一列 1
        self.init_params(x_train.shape[1], y_train.shape[1])
        self._fit_sgd(x_train, y_train, x_test, y_test)

    def _fit_sgd(self, x_train, y_train, x_test, y_test):
        '''
        梯度下降求解，按照批量self.batch_size ，分为随机梯度，批量梯度和小批量梯度法
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        '''
        sample_xy = np.c_[x_train, y_train]
        n_features = x_train.shape[1]
        n_samples, self.n_class = y_train.shape
        for i in range(self.max_epochs):
            self.eta *= 0.95
            np.random.shuffle(sample_xy)
            for idx in range(sample_xy.shape[0] // self.batch_size):
                batch_x_y = sample_xy[idx * self.batch_size:min(n_samples, (idx + 1) * self.batch_size)]
                batch_x, batch_y = batch_x_y[:, :n_features], batch_x_y[:, n_features:]
                # 权重更新增量，softmax激活函数
                y_pred_batch = self.softmax_func(batch_x.dot(self.weight))  # 当前批量预测概率
                dw = ((y_pred_batch - batch_y).T.dot(batch_x) / self.batch_size).T
                # 正则化
                dw_reg = np.zeros(shape=(n_features - 1, self.n_class))  # 正则化不好含偏置项
                if self.l1_ratio:
                    dw_reg += self.l1_ratio * self.sign_func(self.weight[:-1, :]) / self.batch_size
                if self.l2_ratio:
                    dw_reg += 2 * self.l2_ratio * self.weight[:-1, :] / self.batch_size
                dw_reg = np.r_[dw_reg, np.zeros(shape=(1, self.n_class))]
                dw += dw_reg
                self.weight = self.weight - self.eta * dw  # 更新权重
            # 计算交叉熵损失，后期可视化
            y_train_pred = self.softmax_func(x_train.dot(self.weight))  # 当前迭代训练集预测
            self.train_loss.append(self.cal_cross_entropy(y_train, y_train_pred))
            y_test_pred = self.softmax_func(x_test.dot(self.weight))
            self.test_loss.append(self.cal_cross_entropy(y_test, y_test_pred))
            if i > 10 and np.abs(self.train_loss[-1] - self.train_loss[-2]) <= self.eps:
                break

    def get_params(self):
        '''
        :return:
        '''
        if self.fit_intercept:
            w, b = self.weight[:-1, :], self.weight[-1:, :]
        else:
            w, b = self.weight, 0
        if self.normalize:
            w = w / self.feature_std.reshape(-1, 1)
            b = b - w.T.dot(self.feature_mean)
        return w, b

    def predict_prob(self, x_test):
        '''

        :param x_test:
        :return:
        '''
        if self.normalize:
            x_test = (x_test - self.feature_mean) / self.feature_std
        if self.fit_intercept:
            x_test = np.c_[x_test, np.ones(shape=(x_test.shape[0]))]
        y_prob = self.softmax_func(x_test.dot(self.weight))
        return y_prob

    def predict(self, x_test):
        '''

        :param x_test:
        :return:
        '''
        y_prob = self.predict_prob(x_test)
        return np.argmax(y_prob, axis=1)

    def plt_cross_entropy_1oss(self, is_show=False):
        # plt.figure(figsize=(8, 6))
        plt.plot(self.train_loss, 'k--', lw=1, label='Train loss')
        plt.plot(self.test_loss, 'r--', lw=1.2, label='Test loss')
        plt.xlabel('$x$', fontdict={'fontsize': 12})
        plt.ylabel('$y$', fontdict={'fontsize': 12})
        plt.grid(ls=':')
        plt.legend(frameon=False)
        plt.title("train and test ")
        if is_show:
            plt.show()


if __name__ == '__main__':
    iris = load_iris()  # 加载数据集
    X, y = iris.data, iris.target  # 提取样本数据和目标值
    X_train, X_test, y_train, y_test = train_test_split( \
        X, y, test_size=0.3, random_state=0, shuffle=True, stratify=y)
    lgmc = LogisticRegressionMulClass(eta=0.05, \
                                      l1_ratio=0.05, \
                                      l2_ratio=0.05, \
                                      batch_size=5, max_epochs=1000, eps=1e-10)
    lgmc.fit(X_train, y_train, X_test, y_test)
    # 模型训练
    plt.figure(figsize=(12, 8))
    # 可视化，四个子图
    plt.subplot(221)
    lgmc.plt_cross_entropy_1oss(is_show=False)
    # 交叉熵损失下降曲线
    y_test_pred = lgmc.predict(X_test)
    # 模型预测类别
    y_test_prob = lgmc.predict_prob(X_test)
    # 模型预测概率
    feature_names = iris.feature_names
    # 数据集的特征名称
    for fn, theta in zip(feature_names, lgmc.get_params()[0]):  # 输出模型参数
        print(fn, ":", theta)
    print("bias:", lgmc.get_params()[1])  # 偏置项
    print("1" * 100)
    y_test_pred=lgmc.one_hot_encoding(y_test_pred)
    pm = ModelPerformanceMetrics(y_test, y_test_pred)  # 模型性能度量
    print(pm.cal_classification_report())

    plt.subplot(222)
    pr_values = pm.precision_recall_curve()
    pm.plt_pr_curve(pr_values, is_show=False)
    roc_values = pm.roc_metrics_curve()
    plt.subplot(223)
    pm.plt_roc_curve(roc_values)
    cm=pm.cal_confusion_matrix()
    plt.subplot(224)
    lgmc.plt_confusion_matrix(cm,label_names=iris.target_names,is_show=False)
    plt.show()
