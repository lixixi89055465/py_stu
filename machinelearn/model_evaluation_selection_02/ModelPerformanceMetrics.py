# -*- coding: utf-8 -*-
# @Time    : 2023/10/8 11:25
# @Author  : nanji
# @Site    : 
# @File    : ModelPerformanceMetrics.py
# @Software: PyCharm 
# @Comment :
import numpy as np

from sklearn.preprocessing import label_binarize
import pandas as pd


class ModelPerformanceMetrics:
    '''
    性能度量指标，初始参数为y_true 为真实类别标记 ，且为一维数组
    y_prob为预测类别标记，且为二位数组
    '''

    def __init__(self, y_true, y_prob):
        '''
        参数初始化，若为多酚类，分别对y_true进行one-hot编码 ,并默认计算混淆矩阵
        :param y_true:
        :param y_prob:
        '''
        self.y_prob = np.asarray(y_prob, dtype=np.int)
        self.y_true = np.asarray(y_true, dtype=np.int)
        self.n_samples, self.n_class = self.y_prob.shape
        if self.n_class > 2:
            self.y_true = label_binarize(self.y_true, classes=np.unique(self.y_true))
        else:
            self.y_true = self.y_true.reshape(-1)
        self.m = self.cal_confusion_matrix()  #

    def cal_confusion_matrix(self):
        '''
        计算混淆矩阵 ，以预测样本的概率为基准，查询概率最大的索引，即类别

        :return:
        '''
        confusion_matrix = np.zeros((self.n_class, self.n_class), dtype=np.int)
        if self.n_class == 2:
            for i in range(self.n_samples):
                idx = np.argmax(self.y_prob[i, :])
                if idx == self.y_true[i]:
                    confusion_matrix[idx, idx] += 1;  # 判断正确
                else:
                    confusion_matrix[self.y_true[i], idx] += 1  # 判断错误
        else:
            for i in range(self.n_samples):
                idx = np.argmax(self.y_prob[i, :])  # 概率最大的索引，即类被
                idx_true = np.argmax(self.y_true[i, :])
                if idx_true == idx:
                    confusion_matrix[idx, idx] += 1
                else:
                    confusion_matrix[idx_true, idx] += 1
        return confusion_matrix

    @staticmethod
    def __sort_positive__(y_prob):
        idx = np.argsort(y_prob)[::-1]
        return idx

    def cal_classification_report(self, target_names=None):
        '''
        计算类别报告，模拟sklearn中的classification_report
        :param target_names:
        :return:
        '''
        precision = np.diag(self.cm) / np.sum(self.cm, axis=0)  # 查准率
        recall = np.diag(self.cm) / np.sum(self.cm, axis=1)  # 查全率
        f1_score = 2 * precision * recall / (precision + recall)  # F1
        support = np.sum(self.cm, axis=1, dtype=np.int)  # 各类别测试样本数量
        support_all = np.sum(self.cm, dtype=np.int)  # 总测试样本的熟练
        accuracy = np.sum(np.diag(self.cm)) / support_all  # 正确度
        p_m, r_m = precision.mean(), recall.mean()
        macro_avg = [p_m, r_m, 2 * p_m * r_m] / np.sum(p_m + r_m)  # 宏查准率，宏查全率 ,宏 F1
        # 加权查准率，加权查全率,加权F1,以每一类别占总样本的比例为权重系数
        weight = support / support_all
        weight_avg = [np.sum(weight * precision), np.sum(weight * recall), np.sum(weight * f1_score)]
        # 构造分类报告结构
        columns = ['precisoin', 'recall', 'f1_score', 'support']
        metrics1 = pd.DataFrame(np.array([precision, recall, f1_score, support]).T, \
                                columns=columns)
        metrics2 = pd.DataFrame([['', '', '', ''], ['', '', accuracy, support_all],
                                 np.hstack([macro_avg, support_all]), \
                                 np.hstack([weight_avg, support_all])], \
                                columns=columns)
        c_report = pd.concat([metrics1, metrics2], ignore_index=False)
        if target_names is None:  # 类别标签设置，按照类别0、1、2...
            target_names = [str(i) for i in range(self.n_class)]
        else:
            target_names = list(target_names)  # 类别标签设置
        target_names.extend(['', 'accuracy', 'macro_avg', 'weight_avg'])
        c_report.index = target_names
        return c_report

    def precision_recall_curve(self):
        '''
        Precision 与Recall 曲线，各坐标点的计算
        :return:
        '''
        pr_array = np.zeros((self.n_samples, 2))  # 存储查准率，差全率
        if self.n_class == 2:  # 二分类
            idx = self.__sort_positive__(self.y_prob[:, 0])  # 降序排序
            y_true = self.y_true[idx]
            # 针对每个测试样本得分，作为阈值，预测类别，分别机型计算各指标值
            for i in range(self.n_samples):
                tp, fn, tn, fp = self.__cal_sub_metrics__(y_true, i + 1)  # 计算指标
                pr_array[i, :] = tp / (tp + fn), tp / (tp + fp)
        else:
            precision = np.zeros((self.n_samples, self.n_class))
            recall = np.zeros((self.n_samples, self.n_class))
            for k in range(self.n_class):  # 计算每个列别的R，R，F1指标
                idx = self.__sort_positive__(self.y_prob[:, k])  # 降序排序
                y_true_k = self.y_true[idx]
                for i in range(self.n_samples):
                    tp, fn, tn, fp = self.__cal_sub_metrics__(y_true_k, i + 1)
                    precision[i, k] = tp / (tp + fp)  # 查准率
                    recall[i, k] = tp / (tp + fn)  # 查全率
            # 宏查准率Precision和宏查全率Recall
            pr_array = np.array([np.mean(recall), np.mean(precision, axis=1)]).T
        return pr_array

    def __cal_sub_metrics__(self, y_true_sort, n):
        '''
        计算TP，FN，TN，FP，n为第几个样本（从1开始），二分类则假设0为正例，1为负例
        多分类：由于采用了one-host编码，则1为正例，0为负例
        :param y_true_sort: 真值重排数据
        :param n: 以第n个样本概率为阈值
        :return:
        '''
        # 第n个样本预测概率为阈值，则前n个为正例，标记0,后面的皆为范例，1
        if self.n_samples == 2:
            pre_label = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
            tp = pre_label[pre_label == 0 & (pre_label == y_true_sort)]  # 真正例
            tn = pre_label[pre_label == 1 & (pre_label == y_true_sort)]  # 真反例
            fp = np.sum(y_true_sort) - tn  # 假正例
            fn = self.n_samples - tp - tn - fp  # 假反例
        else:
            pre_label = np.r_[np.ones(n, dtype=int), np.zeros(n, dtype=int)]
            tp = pre_label[pre_label == 1 & (pre_label == y_true_sort)]  # 真正例
            tn = pre_label[pre_label == 0 & (pre_label == y_true_sort)]  # 真反例
            fn = np.sum(y_true_sort) - tp  # 假正例
            fp = self.n_samples - tp - tn - fn  # 假反例
        return tp, fn, tn, fp

    def roc_metrics_curve(self):
        '''
        ROC曲线 ,各坐标点的计算
        :return:
        '''
        roc_array = np.zeros((self.n_samples, 2))
