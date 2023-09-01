# -*- coding: utf-8 -*-
# @Time    : 2023/9/1 22:39
# @Author  : nanji
# @Site    : 
# @File    : Performance_metrics.py
# @Software: PyCharm 
# @Comment :


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ModelPerformanceMetrics:
    def __init__(self, y_true, y_prob):
        '''
        初始化参数
        :param y_true:  样本的真实类别
        :param y_prob:  样本的预测类别
        '''
        self.y_true = np.asarray(y_true, dtype=np.int)
        self.y_prob = np.asarray(y_prob, dtype=np.float)
        self.n_samples, self.n_class = self.y_prob.shape  # 样本量和类别数
        if self.n_class > 2:  # 多分类
            self.y_true = self.label_one_hot()
        else:
            self.y_true = self.y_true.reshape(-1)

    def label_one_hot(self):
        '''
        对真实类别标签进行one-hot编码，编码后的维度与模型预测概率维度一致
        :return:
        '''
        y_true_lab = np.zeros((self.n_samples, self.n_class))
        for i in range(self.n_samples):
            y_true_lab[i, self.y_true[i]] = 1
        return y_true_lab

    def cal_confusion_matrix(self):
        '''
        计算并构建混淆矩阵
        :return:
        '''
        confusion_matrix = np.zeros((self.n_class, self.n_class))
        for i in range(self.n_samples):
            idx = np.argmax(self.y_prob[i, :])  # 最大概率所对应的索引，即是类别
            if self.n_class == 2:
                idx_true = self.y_true[i]
            else:
                idx_true = np.argmax(self.y_true[i, :])
            if idx_true == idx:
                confusion_matrix[idx, idx] += 1  # 预测正确，则在对角线元素位置+1
            else:
                confusion_matrix[idx_true, idx] += 1  # 预测错误，则在真实类别行预测错误列+1
        self.cm = confusion_matrix
        return confusion_matrix

    def cal_classification_report(self):
        '''
        计算并构造分类报告
        :return:
        '''
        precision = np.diag(self.cm) / np.sum(self.cm, axis=0)  # 查准率
        recall = np.diag(self.cm) / np.sum(self.cm, axis=1)  # 查全率
        f1_score = 2 * precision * recall / (precision + recall)
        support = np.sum(self.cm, axis=1)  # 各个类别的支持样本量
        support_all = np.sum(support)  # 总的样本量
        p_m, r_m = precision.mean(), recall.mean()
        accuracy = np.sum(np.diag(self.cm)) / support_all  # 准确率
        macro_avg = [accuracy, p_m, r_m, 2 * p_m * r_m / (p_m + r_m)]
        weight = support / support_all  # 以各个类别的样本量所占总的样本量比例为权重。
        weighted_avg = [np.sum(weight * precision), np.sum(weight * recall), np.sum(weight * f1_score)]
        # 构造分类报告
        metrics_1 = pd.DataFrame(np.array([precision, recall, f1_score, support]).T,
                                 columns=['precision', 'recall', 'f1-score', 'support'])
        metrics_2 = pd.DataFrame(
            ['', '', '', ''], ['', '', accuracy, support_all],
            np.hstack([macro_avg, support_all]),
            np.hstack([weighted_avg, support_all])
        )
        print(metrics_1)
