# -*- coding: utf-8 -*-
# @Time    : 2023/9/24 下午1:17
# @Author  : nanji
# @Site    : 
# @File    : entropy_utils.py
# @Software: PyCharm 
# @Comment :
import numpy as np


class EntropyUtils:
    '''
    决策树中各种熵的计算，包括信息熵、信息增益、信息增益率、基尼系数
    统一要求：按照信息增益最大，信息增益率最大、基尼指数增益最大
    '''

    @staticmethod
    def _set_sample_weight(sample_weight, n_sample):
        '''
        扩展到集成学习，此处为样本权重的设置

        :param sample_weight:各样本的权重
        :param n_sample: 样本量
        :return:
        '''
        if sample_weight is None:
            sample_weight = np.asarray([1.0] * n_sample)
        return sample_weight

    def cal_info_entropy(self, y_labels, sample_weight=None):
        '''
        计算样本的信息熵
        :param y_labels: 递归样本子集中类别集合
        :param sample_weight: 各样本的权重
        :return:
        '''
        y = np.asarray(y_labels)
        sample_weight = self._set_sample_weight(sample_weight, len(y))
        y_values = np.unique(y)
        ent_y = 0.0
        for val in y_values:
            p_i = len(y[y == val]) / len(y)
            ent_y += -p_i * np.log2(p_i)
        return ent_y

    def conditional_entropy(self, feature_x, y_labels, sample_weight=None):
        '''
        计算条件熵 ，给定特征属性的情况下，信息熵的计算
        :param feature_x:
        :param y_labels:
        :param sample_weight:
        :return:
        '''
        x, y = np.asarray(feature_x), np.asarray(y_labels)
        sample_weight = self._set_sample_weight(sample_weight, len(y))
        cond_ent = 0.0
        for x_val in np.unique(x):
            x_idx = np.where(x == x_val)
            sub_x, sub_y = x[x_idx], y[x_idx]
            sub_sample_weight = sample_weight[x_idx]
            p_k = len(sub_y) / len(y)
            cond_ent += p_k * self.cal_info_entropy(sub_y, sub_sample_weight)

        return cond_ent

    def info_gain(self, feature_x, y_labels, sample_weight=None):
        '''
        计算信息增益
        :param feature_x:
        :param y_labels:
        :param sample_weight:
        :return:
        '''
        return self.cal_info_entropy(y_labels, sample_weight) - \
               self.conditional_entropy(feature_x, y_labels, sample_weight)


if __name__ == '__main__':
    y = np.random.randint(0, 4, 50)
    entropy = EntropyUtils()
    ent = entropy.cal_info_entropy(y_labels=y)
    print(ent)
