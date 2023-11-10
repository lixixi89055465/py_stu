# -*- coding: utf-8 -*-
# @Time    : 2023/9/14 10:52
# @Author  : nanji
# @Site    : 
# @File    : testSVM_digit.py
# @Software: PyCharm 
# @Comment : 调参与最终模型——SVM+手写数字分类
from hyperopt import fmin, tpe, hp, rand
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split

# 有别于 GridSearch ,只需要给出最优参数的概率分布即可
parameter_space_svc = {
    # loguniform表示该参数取对数后服从均匀分布
    'C': hp.loguniform('C', np.log(1), np.log(100)),
    'kernel': hp.choice('kernel', ['rbf', 'poly']),
    'gamma': hp.loguniform('gamma', np.log(0.001), np.log(0.1))
}
# sklearn.datasets 中内置的手写数字图片数据集
digits = datasets.load_digits()
print(digits.images[0])
# fig = plt.figure(figsize=(15, 8))
# 设置子图形布局
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
# for i in range(24):
#     # 初始化子图；在4*6的网络中，在第i+1个位置添加一个子图
#     ax = fig.add_subplot(4, 6, i + 1, xticks=[], yticks=[])
#     ax.imshow(digits.images[i])  # 在第一个位置上显示图像

# plt.show()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=0)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# pip_svc=make_pipeline(StandardScaler(),PCA(n_components=20),SVC() )
count = 0  # 计数器
scores = []
from sklearn.model_selection import cross_val_score


def function(args):
    clf = SVC(**args)
    score = cross_val_score(clf, X_train, y_train).mean()  # clf可为pipe_svc
    global count
    count += 1
    scores.append(score)
    print('[%d],%s, validate acc: %.5f' % (count, args, score))
    return -score


# algo:
# 1.随机搜索（hyperopt.rand.suggest)2.模拟退火 (hyperopt.anneal.suggest)
# 3. TPC算法（hyperopt.tpe.suggest, Tree-structured Parzen Estimator Approach )
# max_eval 指定枚举次数上限，返回目前搜索道德最优解，不一定是全局最优
best = fmin(function, parameter_space_svc, algo=tpe.suggest, max_evals=100)
# best['kernel']返回的是数组的下标，因此需要把它还原回来
print('aaaaaaaaaa')
kernel_list = ['rbf', 'poly']
best['kernel'] = kernel_list[best['kernel']]
print('best params: ', best)

clf = svm.SVC(**best)  # 根据最佳参数对测试数据进行预测
clf.fit(X_train, y_train)
print(clf.score(X_test,y_test ))
plt.plot(scores,'ko--',markersize=5,markeredgecolor='r')
plt.xlabel('eval times',fontsize=12)
plt.ylabel('score ',fontsize=12)
plt.grid(ls=':')
plt.title('Super parameter tuning of SVM by Hyperopt ',fontsize=14 )
plt.show()
