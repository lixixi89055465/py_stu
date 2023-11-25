from sklearn.datasets import make_classification
import numpy as np
import scipy as sp
from sklearn.preprocessing import StandardScaler


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

# irir = load_iris()
# X, y = irir.data, irir.target
# lda_dr = LDAMultic_DimReduction(n_components=2)
# X_new = lda_dr.fit_transform(X, y)
# print(lda_dr.variance_explained())
# # 可视化
# plt.figure(figsize=(7, 10))
# plt.subplot(211)
# plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
# plt.xlabel('PC1', fontdict={'fontsize': 12})
# plt.ylabel('PC2', fontdict={'fontsize': 12})
# plt.grid(ls=':')
# plt.title('Linear Discriminant Analysis Dimension Reduction ', fontdict={'fontsize': 12})
# # 采用sklearn自带库函数测试
# lda = LinearDiscriminantAnalysis(n_components=2)
# lda.fit(X, y)
# x_lda = lda.transform(X)
# plt.subplot(212)
# plt.scatter(x_lda[:, 0], x_lda[:, 1], marker='s', c=y)
# plt.xlabel('PC1', fontdict={'fontsize': 12})
# plt.ylabel('PC2', fontdict={'fontsize': 12})
# plt.grid(ls=':')
# plt.title('LDA(Sklearn) Dimension Reduction', fontdict={'fontsize': 14})
# plt.show()

X, y = make_classification(n_samples=2000, n_features=20, n_classes=5, \
                           n_informative=3, n_redundant=0, n_repeated=0, \
                           n_clusters_per_class=1, class_sep=2, random_state=42)
X = StandardScaler().fit_transform(X)
lda_dr = LDAMultic_DimReduction(n_components=3)
X_new = lda_dr.fit_transform(X,y)
print(lda_dr.variance_explained())
print('0' * 100)

plt.figure(figsize=(14, 10))
plt.subplot(221)
plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
plt.xlabel('PC1', fontdict={'fontsize': 12})
plt.ylabel('PC2', fontdict={'fontsize': 12})
plt.grid(ls=':')
plt.title('LInear Discriminant Analysis Dimension Reduction', fontdict={'fontsize': 12})

plt.subplot(222)
plt.scatter(X_new[:, 1], X_new[:, 2], marker='o', c=y)
plt.xlabel('PC2', fontdict={'fontsize': 12})
plt.ylabel('PC3', fontdict={'fontsize': 12})
plt.grid(ls=':')
plt.title("Linear Discriminant Analysis Dimension Reduction")

# lda = LinearDiscriminantAnalysis(n_components=3)
# lda.fit(X, y)
# x_lda = lda.transform(X)
# plt.subplot(223)
# plt.scatter(x_lda[:, 0], x_lda[:, 1], marker='s', c=y)
# plt.xlabel('PC1', fontdict={'fontsize': 12})
# plt.ylabel('PC2', fontdict={'fontsize': 12})
# plt.grid(ls=':')
# plt.title('LDA(Sklearn) Dimension Reduction', fontdict={'fontsize': 12})


X,y = make_classification(n_samples=2000, n_features=20, n_informative=3, n_redundant=0, \
                        n_repeated=0, n_classes=5, n_clusters_per_class=1, class_sep=2, )
lda = LinearDiscriminantAnalysis(n_components=3)
lda.fit(X, y)
X_new = lda.transform(X)
plt.subplot(223)
plt.scatter(X_new[:, 0], X_new[:, 1], marker='s', c=y)
plt.xlabel('PC1', fontdict={'fontsize': 12})
plt.ylabel('PC2', fontdict={'fontsize': 12})
plt.grid(ls=':')
plt.title('LDA Dimension Reduction', fontdict={'fontsize': 12})
plt.subplot(224)
x_lda = lda.transform(X)
plt.scatter(x_lda[:, 1], x_lda[:, 2], marker='s', c=y)
plt.show()
