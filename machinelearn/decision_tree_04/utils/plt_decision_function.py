# -*- coding: utf-8 -*-
# @Time    : 2023/9/30 19:57
# @Author  : nanji
# @Site    : 
# @File    : plt_decision_function.py
# @Software: PyCharm 
# @Comment :

import matplotlib.pyplot as plt
import numpy as np


def plot_decision_function(X, y, clf, acc=None, title_info=None, is_show=True, support_vectors=None):
    '''
    可视化分类边界函数
    :param X: 测试样本
    :param y:  分类模型
    :param acc: 模型分类正确率，可不传参数
    :param clf: 分类模型
    :param title_info: 可视化标题title的额外信息
    :param is_show: 是否在当前显示图像，用于夫函数绘制子图
    :param support_vectors: 扩展支持向量机
    :return:
    '''
    if is_show:
        plt.figure(figsize=(7, 5))
    # 根据特征变量的最小值和最大值，生成二维网络，用于绘制等值线
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xi, yi = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    y_pred = clf.predict(np.c_[xi.ravel(), yi.ravel()])  # 模型预测值
    y_pred = y_pred.reshape(xi.shape)
    plt.contourf(xi, yi, y_pred, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolors='k')
    plt.xlabel('Feature 1', fontdict={'fontsize': 12})
    plt.ylabel('Feature 2', fontdict={'fontsize': 12})
    if acc:
        if title_info:
            plt.title("Model Classification Boundary %s \n (accuracy = %.5f)"
                      % (title_info, acc), fontdict={'fontsize': 14})
        else:
            plt.title('Model classification Boundary (accuracy =%.5f )' % acc, fontdict={'fontsize': 14})
    else:
        if title_info:
            plt.title("Model Classification Boundary ", fontdict={'fontsize': 14})
        else:
            plt.title('Model classification Boundary %s' % title_info, fontdict={'fontsize': 14})
    if support_vectors is not None:  # 可视化支持向量，针对SVM
        plt.scatter(X[support_vectors, 0], X[support_vectors, 1], s=80, c='none', alpha=0.7, edgecolors='red')
    if is_show:
        plt.show()
