# -*- coding: utf-8 -*-
# @Time : 2024/1/14 21:41
# @Author : nanji
# @Site : https://www.bilibili.com/video/BV1Ca4y1t7DS?p=13&spm_id_from=pageDriver&vd_source=50305204d8a1be81f31d861b12d4d5cf
# @File : XGBoost01.py
# @Software: PyCharm 
# @Comment :
import time
import numpy as np
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_boston
import matplotlib
import matplotlib.pyplot as plt
import os

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, \
													random_state=1234565)
params = {
	# 通用参数
	'booster': 'gbtree',  # 使用的弱学习器,有两种选择gbtree（默认）和gblinear,gbtree是基于
	# 树模型的提升计算，gblinear是基于线性模型的提升计算
	'nthread': 4,  # XGBoost运行时的线程数，缺省时是当前系统获得的最大线程数
	'silent': 0,  # 0：表示打印运行时信息，1：表示以缄默方式运行，默认为0
	'num_feature': 4,  # boosting过程中使用的特征维数
	'seed': 1000,  # 随机数种子
	# 任务参数
	'objective': 'multi:softmax',  # 多分类的softmax,objective用来定义学习任务及相应的损失函数
	'num_class': 3,  # 类别总数
	# 提升参数
	'gamma': 0.1,  # 叶子节点进行划分时需要损失函数减少的最小值
	'max_depth': 6,  # 树的最大深度，缺省值为6，可设置其他值
	'lambda': 2,  # 正则化权重
	'subsample': 0.7,  # 训练模型的样本占总样本的比例，用于防止过拟合
	'colsample_bytree': 0.7,  # 建立树时对特征进行采样的比例
	'min_child_weight': 3,  # 叶子节点继续划分的最小的样本权重和
	'eta': 0.1,  # 加法模型中使用的收缩步长

}
