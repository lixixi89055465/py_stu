import numpy as np
import pandas as pd

wdbc = pd.read_csv('../breast+cancer+wisconsin+diagnostic/wdbc.data')
# print(wdbc.info())
from sklearn.preprocessing import LabelEncoder, StandardScaler  # 类别标签，标准化处理

X, y = wdbc.iloc[:, 2:].values, wdbc.iloc[:, 1]
# print(X)
# print('0' * 100)
# print(y)
# print('1' * 100)
# print(X[1, :])
X = StandardScaler().fit_transform(X)
# print('2' * 100)
# print(X[1, :])
lab_en = LabelEncoder()  # 对目标值进行编码，常见对象
y = lab_en.fit_transform(y)  # 拟合和转换
# print('3' * 100)
# print(y)
print('4' * 100)
# print(lab_en.classes_)
# lab_en.transform(["B", "M"])
print('5' * 100)
print(lab_en.classes_)
# print('6' * 100)

# 降噪，降维，可视化
from sklearn.decomposition import PCA  # 主成分分析

pca = PCA(n_components=6).fit(X)  # 选取6个主成分保留  30维=》6维
evr = pca.explained_variance_ratio_  # 解释方差比，即各个主成分的贡献
print("各个主成分贡献率", evr, "\n累计贡献率", np.cumsum(evr))  # 信息损失了大概11%
print('7' * 100)
print(evr)
X_pca = pca.transform(X)  # 转换获取各个主成分
print('0' * 100)
print(X_pca.shape)
X_b, X_m = X_pca[y == 0], X_pca[y == 1]
print('1' * 100)
print(X_b.shape)
print(X_m.shape)
import matplotlib.pyplot as plt

# plt.figure(figsize=(21, 5))
# for i in range(3):
#     plt.subplot(131 + i)
#     plt.plot(X_b[:, i * 2], X_b[:, i * 2 + 1], "ro")
#     plt.plot(X_m[:, i * 2], X_m[:, i * 2 + 1], "b+")
# plt.show()

from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn.metrics import classification_report, accuracy_score  # 分类报告，正确率

X_train, X_test, y_train, y_tes = train_test_split(X_pca, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)
print(len(X_test), len(y_train))

log_reg = LogisticRegression()  # 未掌握原理之前，所有参数默认
log_reg.fit(X_train)

