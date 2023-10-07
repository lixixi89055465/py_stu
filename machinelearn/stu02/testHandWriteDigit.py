# -*- coding: utf-8 -*-
# @Time    : 2023/9/26 16:19
# @Author  : nanji
# @Site    : 
# @File    : testHandWriteDigit.py
# @Software: PyCharm 
# @Comment :3. 性能度量——逻辑回归+手写数字分类手写数字分类

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
X, y = digits.data, digits.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True, stratify=y)

# 采用逻辑回归进行多酚类
from sklearn.linear_model import LogisticRegression

model_lg = LogisticRegression()  # 逻辑回归模型实力
model_lg.fit(X_train, y_train)  # 训练样本
y_test_pred = model_lg.predict(X_test)  # 测试样本预测
# 单独计算性能指标
from sklearn import metrics
# test_accuracy=metrics.accuracy_score(y_test,y_test_pred)
# print(test_accuracy)
# print('0'*100)
# macro=metrics.precision_score(y_test,y_test_pred,average="macro")
# print(macro)
# print('1'*100)
# micro=metrics.recall_score(y_test,y_test_pred,average="micro")
# print(micro)
# print('2'*100)
# f1_weight=metrics.f1_score(y_test,y_test_pred,average='micro')
# print(f1_weight)
# print('3'*100)
# macro=metrics.f1_score(y_test,y_test_pred,average='macro')
# print(macro)
# print('4'*100)
# f1_weighted=metrics.f1_score(y_test,y_test_pred,average='weighted')
# print(f1_weighted)
# print('5'*100)
# fbeta=metrics.fbeta_score(y_test,y_test_pred,average='macro',beta=1)
# print(fbeta)
# 绘制混淆矩阵
import matplotlib.pyplot as plt

# fig,ax=plt.subplots(figsize=(10,8))
# target_names=[] # 用来命名类别
# for i in range(10):
#     target_names.append('n'+str(i))

# plot_confusion_matrix(model_lg,X_test,y_test,display_labels=target_names,cmap=plt.cm.Reds,ax=ax)
# plt.show()

# print('-'*100)
# print(metrics.confusion_matrix(y_test, y_test_pred))
# print('-'*100)
# print(metrics.classification_report(y_test, y_test_pred, target_names=target_names))

# print('4'*100)
# prfs=metrics.precision_recall_fscore_support(y_test,y_test_pred,beta=1,average=None)
# print(prfs)
cm = metrics.confusion_matrix(y_test, y_test_pred)
print(cm)
import numpy as np

print('0' * 100)
precision = np.diag(cm) / np.sum(cm, axis=0)
print(precision)
recall = np.diag(cm) / np.sum(cm, axis=1)
print('1' * 100)
print(recall)
f1_score = 2 * recall * precision / (recall + precision)
print(f1_score)
support = np.sum(cm, axis=1)  # 各类别支持样本量
print('2' * 100)
print(support)
print('3' * 100)
print(np.sum(support))  # 总样本量
print('4' * 100)
accuracy = np.sum(np.diag(cm)) / np.sum(cm)  # 精度
print(accuracy)

# 宏查准率 ,宏召回率，宏-F1
macro_avg = [precision.mean(), recall.mean(),
             2 * precision.mean() * recall.mean() / (precision.mean() + recall.mean())]
# 加权查准率，加权召回率，加权-F1
support_all = np.sum(support)
weight = support / support_all
weight_avg = [np.sum(weight * precision), np.sum(weight * recall),
              (np.sum(weight * f1_score))]
import pandas as pd
metrics1=pd.DataFrame(np.array([precision,recall,f1_score,support]).T,
                      columns=['precision','recall','f1_score','support'])

metrics2=pd.DataFrame([['','','',''],['','',accuracy,support_all],
                       np.hstack([macro_avg]),
                       np.hstack([weight_avg,support_all])],
                      columns=['precision','recall','f1_score','support'])
metrics=pd.concat([metrics1,metrics2],ignore_index=True)
target_names=[]# 用来命名类别
for i in range(10):
    target_names.append("n"+str(i))
target_names.extend([''])
metrics.index=target_names
print('2'*100)
print(metrics)
