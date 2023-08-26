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

acc_test_score, acc_train_score = [], []  # 每次随机划分训练和测试评分
for i in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=i, shuffle=True,
                                                        stratify=y)
    # print(len(X_test), len(y_train))
    log_reg = LogisticRegression()  # 未掌握原理之前，所有参数默认
    log_reg.fit(X_train, y_train)  # 采用训练集训练模型
    y_test_pred = log_reg.predict(X_test)  # 模型训练完毕后，对测试样本进行预测
    # print((y_test_pred == y_test).sum() * 1.0 / len(y_test))  #
    accuracy_score(y_test, y_test_pred)
    acc_train_score.append(accuracy_score(y_train, log_reg.predict(X_train)))
    acc_test_score.append(accuracy_score(y_test, log_reg.predict(X_test)))

print(acc_train_score)
print('0' * 100)
print(acc_test_score)
# plt.figure(figsize=(7, 5))
# plt.plot(acc_test_score, "ro:", lw=1.5, label='Test')
# plt.plot(acc_train_score, "ks--", lw=1, markersize=4, label='Test')
# plt.show()

from sklearn.pipeline import Pipeline, make_pipeline

pipe_lr = make_pipeline(
    StandardScaler(),
    PCA(n_components=6),
    LogisticRegression()
)
X, y = wdbc.iloc[:, 2:].values, wdbc.iloc[:, 1]
y = LabelEncoder().fit_transform(y)  # 对目标值进行编码
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y)
pipe_lr.fit(X_train, y_train)
pipe_lr.predict(X_test)
print('2' * 100)
# print(accuracy_score(X_test, y_test))
print('Test accuracy is %.5f' % pipe_lr.score(X_test, y_test))

from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
scores = []
for i, (train_idx, test_idx) in enumerate(kfold):
    pipe_lr.fit(X_train[train_idx], y_train[train_idx])
    score = pipe_lr.score(X_train[test_idx], y_train[test_idx])
    scores.append(score)
    print('Fold %2d的,class %s,Acc:%.3f' % (i + 1, np.bincount(y_train[train_idx]), score))
print('CV,%.3f, %.3f' % (np.sum(scores) / len(scores), np.std(scores)))

print('1' * 100)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=-1)
print(len(scores))
print(scores)
print('2' * 100)
from sklearn.neighbors import KNeighborsClassifier

pipe_knn = make_pipeline(
    StandardScaler(),  # 标准化，数据保留相同的数量级
    PCA(n_components=10),  # 主成分分析，降维
    KNeighborsClassifier())  # 具体模型：K近邻算法
# pipe_knn.fit(X_train, y_train)
# y_predict = pipe_knn.transform(X_test)
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, shuffle=True)
# scores = cross_val_score(estimator=pipe_lr, x=X_train, y=y_train, cv=10, n_jobs=-1)
print(score)
k_range = range(1, 3)  # k值选择范围，其他参数默认
# cv_scores = []  # 用于存储每个k值的10折交叉验证均分
print('3' * 100)
print(pipe_knn)
print('4' * 100)
cv_scores = []
for k in k_range:
    pipe_knn.set_params(kneighborsclassifier__n_neighbors=k)  # 设置最近邻参数
    scores = cross_val_score(estimator=pipe_knn, X=X_train, y=y_train, cv=10, n_jobs=-1, )
    cv_scores.append(np.mean(scores))

print(cv_scores)
print('5' * 100)
print(KNeighborsClassifier)
plt.figure(figsize=(8, 6))
plt.plot(k_range, cv_scores, 'ko-', lw=1, markeredgecolor='r')
plt.legend(frameon=False)
plt.grid(ls=':')
plt.xlabel('random division times:',fontsize=12)
plt.ylabel('Accuracy score of test vs train :',fontsize=12)
plt.title('Test samples accuracy score mean= %.5f(+/-%.5f)'%(np.mean(acc_test_score),np.std(acc_test_score)),fontsize=12)
