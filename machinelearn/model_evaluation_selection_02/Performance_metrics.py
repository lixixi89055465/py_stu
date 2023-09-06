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
        计算并构建混淆矩阵(重点 )
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

    def cal_classification_report(self, target_names=None):
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
        macro_avg = [p_m, r_m, 2 * p_m * r_m / (p_m + r_m)]
        weight = support / support_all  # 以各个类别的样本量所占总的样本量比例为权重。
        weighted_avg = [np.sum(weight * precision), np.sum(weight * recall), np.sum(weight * f1_score)]
        # 构造分类报告
        metrics_1 = pd.DataFrame(np.array([precision, recall, f1_score, support]).T,
                                 columns=['precision', 'recall', 'f1-score', 'support'])
        metrics_2 = pd.DataFrame(
            [['', '', '', ''], ['', '', accuracy, support_all],
             np.hstack([macro_avg, support_all]),
             np.hstack([weighted_avg, support_all])],
            columns=['precision', 'recall', 'f1-score', 'support']
        )
        c_report = pd.concat([metrics_1, metrics_2], ignore_index=True)
        if target_names is None:
            target_names = [str(i) for i in range(self.n_class)]
        else:
            target_names = list(target_names)
        target_names.extend(['', 'accuracy', 'macro avg', 'weighted avg'])
        c_report.index = target_names
        c_report = target_names
        return c_report

    @staticmethod
    def __sort_postive__(y_prob):
        '''
        按照预测为正例的概率进行降序排序，并返回排序的索引向量
        :param y_prob:  一维数组，样本预测为正例的概率
        :return:
        '''
        idx = np.argsort(y_prob)[::-1]  # 降序排序
        return idx

    def precision_recall_curve(self):
        '''
        Precision 和 Recall 曲线，计算各坐标点的值，可视化P-R曲线
        :return:
        '''
        pr_array = np.zeros((self.n_samples, 2))  # 存储每个样本预测概率作为阈值时的Phenomen
        if self.n_class == 2:
            idx = self.__sort_postive__(self.y_prob[:, 0])
            y_true = self.y_true[idx]  # 真值类别标签按照排序索引进行排序
            # 针对每个样本，把预测概率作为阈值，计算各指标
            for i in range(self.n_samples):
                tp, fn, tn, fp = self.__call_sub_metrics__(y_true, i + 1)
                pr_array[i, :] = tp / (tp + fn), tp / (tp + fp)
        else:  # 多分类
            precision = np.zeros((self.n_samples, self.n_class))  # 查准率
            recall = np.zeros((self.n_samples, self.n_class))  # 查全率
            for k in range(self.n_class):  # 针对每个类别，分别计算P，R指标，然后平局　
                idx = self.__sort_postive__(self.y_prob[:, k])
                y_true_k = self.y_true[:, k]  # 真值类别第K列
                y_true = y_true_k[idx]  # 对第K 个类别的真值排序
                # 针对每个样本，把预测概率作为阈值，计算各个指标
                for i in range(self.n_samples):
                    tp, fn, tn, fp = self.__call_sub_metrics__(y_true, i + 1)
                    precision[i, k] = tp / (tp + fp)  # 查准率
                    recall[i, k] = tp / (tp + fn)
            # 宏查准率与宏查全率
            pr_array = np.array([np.mean(recall, axis=1), np.mean(precision, axis=1)]).T
        return pr_array

    def roc_metrics_curve(self):
        '''
        ROC 曲线，计算真正利率和假正利率，并可视化
        :return:
        '''
        roc_array = np.zeros((self.n_samples, 2))  # 存储每个样本预测概率作为阈值时的TPR和FPR 指标
        if self.n_class == 2:  # 二分类
            idx = self.__sort_postive__(self.y_prob[:, 0])
            y_true = self.y_true[idx]  # 真值类别标签按照排序索引进行排序
            # 针对每个样本，把预测概率作为阈值，计算各指标
            n_nums, p_nums = len(y_true[y_true == 1]), len(y_true[y_true == 0]) # 真实类别中，反例与正例的样本量
            tp, fn, tn, fp = self.__call_sub_metrics__(y_true,1)
            roc_array[0, :] = fp / (tp + fp), tp / (tp + fn)
            for i in range(self.n_samples):
                if y_true[i]==1:
                    roc_array[i,:]=roc_array[i-1,0]+1/n_nums,roc_array[i-1,1]
                else:
                    roc_array[i,:]=roc_array[i-1,0],roc_array[i-1,1]+1/p_nums
        else:  # 多分类
            fpr = np.zeros((self.n_samples, self.n_class))  # 假正例率
            tpr = np.zeros((self.n_samples, self.n_class))  # 真正例率
            for k in range(self.n_class):  # 针对每个类别，分别计算tpr，fpr指标，然后平局　
                idx = self.__sort_postive__(self.y_prob[:, k])
                y_true_k = self.y_true[:, k]  # 真值类别第K列
                y_true = y_true_k[idx]  # 对第K 个类别的真值排序
                # 针对每个样本，把预测概率作为阈值，计算各个指标
                for i in range(self.n_samples):
                    tp, fn, tn, fp = self.__call_sub_metrics__(y_true, i + 1)
                    fpr[i, k] = tp / (tp + fp)  # 假正例率
                    tpr[i, k] = tp / (tp + fn) # 真正例率
            # 宏查准率与宏查全率
            roc_array = np.array([np.mean(tpr, axis=1), np.mean(fpr, axis=1)]).T
        return roc_array

    def fnr_fpr_metrics_curve(self):
        '''
        代价曲线指标，假反利率，假正例率FPR
        :return:
        '''
        fpr_fnr_array = self.roc_metrics_curve()  # 获取假正例和正真例率
        fpr_fnr_array[:, 1] = 1 - fpr_fnr_array[:, 1]  # 计算假反例率
        return fpr_fnr_array
    def plt_cost_curve(self,fnr_fpr_vals,alpha,class_i=0):
        '''
        可视化代价曲线
        :param fnr_fpr_vals: 假反利率和假正利率二维数组
        :param alpha: alpha=cost10/cost01,更侧重于正例预测为反例的代价，让cost01=1
        :param class_i: 指定绘制第i个类别的代价曲线，如果是二分类，则为0
        :return:
        '''
        plt.figure(figsize=(7,5))
        fpr_s,fnr_s=fnr_fpr_vals[:,0],fnr_fpr_vals[:,1]#获取假正例率和假反利率
        cost01,cost10=1,alpha
        if self.n_class==2:
            class_i=0 # 二分类，默认取第一列
        if 0<=class_i<self.n_class:
            p=np.sort(self.y_prob[:,class_i])
        else:
            p=np.sort(self.y_prob[:,0])# 不满足条件，默认第一个类别
        positive_cost=p*cost01/(p*cost01+(1-p)*cost10)
        for fpr,fnr in zip(fpr_s,fnr_s):
            cost_norm=fnr*positive_cost+(1-positive_cost)*fpr
            plt.plot(positive_cost,cost_norm,'b-',lw=0.5)
        # 查找公共边界，计算期望总体代价
        public_cost=np.outer(fnr_s,positive_cost)+np.outer(fpr_s,(1-positive_cost))
        print(public_cost)
        cost_area=0
        plt.xlabel('Positive Probability cost',fontdict={'fontsize':12})
        plt.ylabel('Normalized Cost',fontdict={'fontsize':12})
        plt.show()








        pass

    @staticmethod
    def __cal_auc__(roc_val):
        '''
        计算ROC曲线下的面积,即AUC
        :param roc_val:
        :return:
        '''
        return (roc_val[1:, 0] - roc_val[:-1, 0]).dot(roc_val[:-1, 1] - roc_val[1:, 1]) / 2

    def plt_roc_curve(self, roc_val, label=None, is_show=None):
        '''
        可视化ROC曲线　
        :param roc_val:  ROC指标各坐标点的数组
        :return:
        '''
        ap = self.__cal_auc__(roc_val)
        plt.figure(figsize=(7, 5))
        if label:
            plt.step(roc_val[:, 0], roc_val[:, 1], '-', lw=2, where='post', label=label + ', AP = %.3f' % ap)
        else:
            plt.step(roc_val[:, 0], roc_val[:, 1], '-', lw=2, where='post', label='AP=%.3f' % ap)
        # plt.figure(figsize=(7, 5))
        plt.title("ROC Curve of Test saples by Different Models")
        # plt.xlabel('Recall', fontdict={'fontsize': 12})
        # plt.ylabel('Precision', fontdict={'fontsize': 12})
        plt.xlabel('False Positive Rate', fontdict={'fontsize': 12})
        plt.ylabel('True Positive Rate', fontdict={'fontsize': 12})
        plt.grid(ls=':')
        plt.legend(frameon=False)  # 添加图例，且曲线图例边框线
        # plt.legend(labels=['频次'])
        # plt.legend(loc=4, bbox_to_anchor=(1.15, -0.07))  # 原代码报错并不显示图例
        # plt.legend(loc=4, bbox_to_anchor=(1.15, -0.07), labels=['频次'])  # 调整后不报错并显示图例
        if is_show:
            plt.show()

    def __call_sub_metrics__(self, y_true_sort, n):
        '''
        计算 TP，TN，FP，TN
        :param y_true_sort: 排序后的真是类别
        :param n: 以第n个样本预测概率为阈值
        :return:
        '''
        if self.n_class == 2:
            pre_label = np.r_[np.zeros(n, dtype=np.int), np.ones(self.n_samples - n, dtype=np.int)]
            tp = len(pre_label[(pre_label == 0) & (pre_label == y_true_sort)])  # 真正例
            tn = len(pre_label[(pre_label == 1) & (pre_label == y_true_sort)])  # 真反例
            fp = np.sum(y_true_sort) - tn  # 假反例
            fn = self.n_samples - tp - tn - fp  # 假正例
        else:
            pre_label = np.r_[np.ones(n, dtype=np.int), np.zeros(self.n_samples - n, dtype=np.int)]
            tp = len(pre_label[(pre_label == 1) & (pre_label == y_true_sort)])  # 真正例
            tn = len(pre_label[(pre_label == 0) & (pre_label == y_true_sort)])  # 真反例
            fn = np.sum(y_true_sort) - tp  # 假反例
            fp = self.n_samples - tp - tn - fn  # 假正例
        return tp, fn, tn, fp

    @staticmethod
    def __cal_ap__(pr_val):
        '''
        计算AP
        :param pr_val:
        :return:
        '''
        return np.dot(pr_val[1:, 0] - pr_val[0:-1, 0], pr_val[1:, 1])

    def plt_pr_curve(self, pr_val, label=None, is_show=None):
        '''
        可视化PR曲线　
        :param pr_val:
        :return:
        '''
        ap = self.__cal_ap__(pr_val)
        plt.figure(figsize=(7, 5))
        if label:
            plt.step(pr_val[:, 0], pr_val[:, 1], '-', lw=2, where='post', label=label + ', AP = %.3f' % ap)
        else:
            plt.step(pr_val[:, 0], pr_val[:, 1], '-', lw=2, where='post', label='AP=%.3f' % ap)
        # plt.figure(figsize=(7, 5))
        plt.title("title")
        # plt.xlabel('Recall', fontdict={'fontsize': 12})
        # plt.ylabel('Precision', fontdict={'fontsize': 12})
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(ls=':')
        plt.legend(frameon=False)  # 添加图例，且曲线图例边框线
        # plt.legend(labels=['频次'])
        # plt.legend(loc=4, bbox_to_anchor=(1.15, -0.07))  # 原代码报错并不显示图例
        # plt.legend(loc=4, bbox_to_anchor=(1.15, -0.07), labels=['频次'])  # 调整后不报错并显示图例
        if is_show:
            plt.show()
