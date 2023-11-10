# -*- coding: utf-8 -*-
# @Time    : 2023/10/8 11:25
# @Author  : nanji
# @Site    : 
# @File    : ModelPerformanceMetrics.py
# @Software: PyCharm 
# @Comment :
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import label_binarize
import pandas as pd
import matplotlib.pyplot as plt


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
        self.y_prob = np.asarray(y_prob, dtype=np.float)
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
        self.cm=confusion_matrix
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
                y_true_k = self.y_true[:, k]  # 取第K 列，即第K个类别所有目标值
                y_true = y_true_k[idx]  # 真值按照修改索引进行重排
                # 针对每个测试样本呢得分，作为阈值，预测类别，并分别计算各目标值
                for i in range(self.n_samples):
                    tp, fn, tn, fp = self.__cal_sub_metrics__(y_true, i + 1)
                    precision[i, k] = tp / (tp + fp)  # 查准率
                    recall[i, k] = tp / (tp + fn)  # 查全率
            # 宏查准率Precision和宏查全率Recall
            pr_array = np.array([np.mean(recall, axis=1), np.mean(precision, axis=1)]).T
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
        if self.n_class == 2:
            pre_label = np.r_[np.zeros(n, dtype=int), np.ones(self.n_samples - n, dtype=int)]
            tp = len(pre_label[pre_label == 0 & (pre_label == y_true_sort)])  # 真正例
            tn = len(pre_label[pre_label == 1 & (pre_label == y_true_sort)])  # 真反例
            fp = np.sum(y_true_sort) - tn  # 假正例
            fn = self.n_samples - tp - tn - fp  # 假反例
        else:
            pre_label = np.r_[np.ones(n, dtype=int), np.zeros(self.n_samples - n, dtype=int)]
            tp = len(pre_label[(pre_label == 1) & (pre_label == y_true_sort)])  # 真正例
            tn = len(pre_label[(pre_label == 0) & (pre_label == y_true_sort)])  # 真反例
            fn = np.sum(y_true_sort) - tp  # 假正例
            fp = self.n_samples - tp - tn - fn  # 假反例
        return tp, fn, tn, fp

    def roc_metrics_curve(self):
        '''
        ROC曲线 ,各坐标点的计算
        :return:
        '''
        roc_array = np.zeros((self.n_samples, 2))  # 存储假正例，真正例率
        if self.n_class == 2:  # 二分类
            idx = self.__sort_positive__(self.y_prob[:, 0])  # 降序排序
            y_true = self.y_true[idx]  # 真值按照排序索引进行重排
            # # 针对每个测试样本得分，作为阈值，预测类别，并分别计算个指标值
            for i in range(self.n_samples):
                tp, fn, tn, fp = self.__cal_sub_metrics__(y_true, i + 1)  # 计算指标
                # 计算假正利率FPR,和真正利率 TPR
                roc_array[i, :] = fp / (fp + tn), tp / (tp + fn)
        else:  # 多分类
            fpr = np.zeros((self.n_samples, self.n_class))
            tpr = np.zeros((self.n_samples, self.n_class))
            for k in range(self.n_class):
                idx = self.__sort_positive__(self.y_prob[:, k])
                y_true_k = self.y_true[:, k]
                y_true = y_true_k[idx]  # 真值按照排序索引进行重排
                # 针对每个测试样本得分，作为阈值，预测类别，并分别计算各指标值
                for i in range(self.n_samples):
                    tp, fn, tn, fp = self.__cal_sub_metrics__(y_true, i + 1)  # 计算指标
                    fpr[i, k], tpr[i, k] = fp / (fp + tn), tp / (tp + fn)
            # 宏假正利率FPR 和宏真正利率TPR
            roc_array = np.array([np.mean(fpr, axis=1), np.mean(tpr, axis=1)]).T
        return roc_array

    def cost_metrics_curve(self, cost_vals):
        '''
        代价曲线，各坐标点的计算
        :param cost_vals:
        :return:
        '''
        cost_array = np.zeros((self.n_samples, 2))  # 存储假正例率代价和归一化代价
        if self.n_class == 2:  # 二分类
            idx = self.__sort_positive__(self.y_prob[:, 0])  # 降序排序
            y_prob = self.y_prob[idx, 0]  # 降序排序，取第一列即可
            y_true = self.y_true[idx]  # 真值按照排序索引进行重排
            # # 针对每个测试样本得分，作为阈值，预测类别，并分别计算个指标值
            cost01, cost10 = cost_vals[0], cost_vals[1]
            # 针对每个测试样本的得分，作为阈值，预测类别，并分别计算各指标值
            for i in range(self.n_samples):
                tp, fn, tn, fp = self.__cal_sub_metrics__(y_true, i + 1)  # 计算指标
                # 计算假正利率FPR,和真正利率 TPR
                p_cost = y_prob[i] * cost01 / (y_prob[i] * cost01 + (1 - y_prob[i]) * cost10)
                fpr, tpr = fp / (fp + tn), tp / (tp + fn)
                fnr = 1 - tpr
                cost_norm = fnr * y_prob[i] + fpr * (1 - y_prob[i])
                cost_array[i, :] = p_cost, cost_norm
        else:
            p_cost = np.zeros((self.n_samples, self.n_class))
            cost_norm = np.zeros((self.n_samples, self.n_class))
            for k in range(self.n_class):  # 计算每个类别的正利率代价和归一化代价
                idx = self.__sort_positive__(self.y_prob[:, k])
                y_prob = self.y_prob[idx, k]
                y_true_k = self.y_prob[:, k]
                y_true = y_true_k[idx]  # 真值按照排序索引进行重排
                cost01, cost10 = cost_vals[k], 1 - cost_vals[k]
                # 针对每个测试样本得分，作为阈值，预测类别，并分别计算各指标值
                for i in range(self.n_samples):
                    tp, fn, tn, fp = self.__cal_sub_metrics__(y_true, i + 1)  # 计算指标
                    p_cost[i, k] = y_prob[i] * cost01 / (y_prob[i] * cost01 + (1 - y_prob[i]) * cost10)
                    tpr, fpr = tp / (tp + fn), fp / (tn + fp)
                    fnr = 1 - tpr  # 假反利率
                    cost_norm[i, k] = fnr * y_prob[i] + fpr * (1 - y_prob[i])  # 归一化代价
            # 宏假正利率FPR 和宏真正利率TPR
            cost_array = np.array([np.mean(p_cost, axis=1), np.mean(cost_norm, axis=1)]).T
        return cost_array

    def plt_pr_value(self, pr_val, label=None, is_show=True):
        '''
        可视化PR 曲线
        :param pr_val:  PR 值数组
        :param label:  用于可视化多个模型的label 图例
        :param is_show: 用于子图的绘制
        :return:
        '''
        ap = self.__cal_ap__(pr_val)  # 计算PR曲线面积
        if is_show:
            plt.figure(figsize=(7, 5))
        if label:
            plt.step(pr_val[:, 0], pr_val[:, 1], '-', lw=2, where='post', label=label + ', AP = %.3f' % ap)
            plt.legend(frameon=False)
            plt.title("Precision recall curve of test samples by different Model")
        else:
            plt.step(pr_val[:, 0], pr_val[:, 1], ls='-', lw=2, where='post')
            plt.title('Precision-Recall curve of test AP = %.3f' % ap)
        plt.xlabel('Recall ', fontdict={'fontsize': 12})
        plt.ylabel('Precision ', fontdict={'fontsize': 12})
        plt.grid(ls=':')
        if is_show:
            plt.show()

    def plt_roc_curve(self, roc_val, label=None, is_show=None):
        '''
        可视化ROC曲线
        :param roc_val:
        :param label:
        :param is_show:
        :return:
        '''
        ap = self.__cal_auc(roc_val)
        plt.plot(roc_val[0], roc_val[1], 'r--', lw=1)  # 公共边界
        if is_show:
            plt.figure(figsize=(7, 5))
        if label:
            plt.step(roc_val[:, 0], roc_val[:, 1], '-', lw=2, where='post', label=label + ', AP = %.3f' % ap)
            plt.legend(frameon=False)
            plt.title("Precision recall curve of test samples by different Model")
        else:
            plt.step(roc_val[:, 0], roc_val[:, 1], ls='-', lw=2, where='post')
            plt.title('Precision-Recall curve of test AP = %.3f' % ap)
        plt.xlabel('Recall ', fontdict={'fontsize': 12})
        plt.ylabel('Precision ', fontdict={'fontsize': 12})
        plt.grid(ls=':')
        if is_show:
            plt.show()

    def __cal_auc(self, roc_val):
        '''
        计算ROC曲线下的面积，即AUC
        :param roc_val:
        :return:
        '''
        return (roc_val[1:, 0] - roc_val[:-1, 0]).dot(roc_val[1:, 1] + roc_val[:-1, 1]) / 2

    def fnr_fpr_metrics_curve(self):
        '''
        代价曲线指标，加反利率，假证利率 FPR
        :return:
        '''
        fpr_fnr_array = self.roc_metrics_curve()  # 获取假证利率和真正利率
        fpr_fnr_array[:, 1] = 1 - fpr_fnr_array[:, 1]  # 计算假反利率
        return fpr_fnr_array

    def plt_cost_curve(self, fnr_fpr_vals, alpha, class_i=0):
        '''
        可视化代价曲线
        :param fnr_fpr_vals: 假反利率和假正利率二维数组
        :param alpha: alpha=cost10/cost01,更侧重于正例预测为反例的代价，让cost01=1
        :param class_i: 指定绘制第i个类别的代价曲线，如果是二分类，则为0
        :return:
        '''
        # plt.figure(figsize=(7,5))
        fpr_s, fnr_s = fnr_fpr_vals[:, 0], fnr_fpr_vals[:, 1]  # 获取假正例率和假反利率
        cost01, cost10 = 1, alpha
        if self.n_class == 2:
            class_i = 0  # 二分类，默认取第一列
        if 0 <= class_i < self.n_class:
            p = np.sort(self.y_prob[:, class_i])
        else:
            p = np.sort(self.y_prob[:, 0])  # 不满足条件，默认第一个类别
        positive_cost = p * cost01 / (p * cost01 + (1 - p) * cost10)
        for fpr, fnr in zip(fpr_s, fnr_s):
            # cost_norm=fnr*positive_cost+(1-positive_cost)*fpr
            # plt.plot(positive_cost,cost_norm,'b-',lw=0.5)
            plt.plot([0, 1], [fpr, fnr], 'b-', lw=0.5)
        # 查找公共边界，计算期望总体代价
        public_cost = np.outer(fnr_s, positive_cost) + np.outer(fpr_s, (1 - positive_cost))
        public_cost_min = public_cost.min(axis=0)
        plt.plot(positive_cost, public_cost_min, 'r--', lw=1)  # 公共边界
        plt.fill_between(positive_cost, 0, public_cost_min, facecolor='g', alpha=0.5)

        cost_area = self.__cal_etc__(positive_cost, public_cost_min)
        plt.xlabel('Positive Probability cost', fontdict={'fontsize': 12})
        plt.ylabel('Normalized Cost', fontdict={'fontsize': 12})
        plt.title('Nequal Cost Curve and Expected Total Cost=%.8f' % cost_area)
        plt.show()

    @staticmethod
    def __cal_etc__(p_cost, cost_norm):
        '''
        计算期望总体代价，即代价曲线公共下线所围城的面积
        :return:
        '''
        return (p_cost[1:] - p_cost[:-1]).dot((cost_norm[:-1] + cost_norm[1:]) / 2)

    def __cal_ap__(self, pr_val):
        return np.dot(pr_val[1:, 0] - pr_val[:-1, 0], pr_val[1:, 1])
