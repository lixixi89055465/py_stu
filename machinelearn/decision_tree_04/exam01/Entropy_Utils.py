import numpy as np
import math


class Entropy_Utils:
    '''
    决策树中各种墒的计算，统一要求：信息增益、增益率和基尼指数增益均按
    最大值选择划分属性
    '''

    def cal_info_entropy(self, y_labels, sample_weight=None):
        '''
        计算信息熵
        :param y_labels:
        :param sample_weight:
        :return:
        '''
        y = np.asarray(y_labels)
        sample_weight = self._set_sample_weight(sample_weight, len(y))
        y_values = np.unique(y)
        ent_x = 0  # 计算信息熵
        for val in y_values:
            p_i = 1.0 * len(y[y == val]) * np.mean(sample_weight) / len(y)
            ent_x += p_i * np.log2(p_i)
        return ent_x

    def _set_sample_weight(self, sample_weight, n_samples):
        if sample_weight is None:
            sample_weight = np.asarray([1.0] * n_samples)
        return sample_weight

    def conditional_entropy(self, feature_x, y_labels, sample_weight):
        '''计算条件熵 H(y|x),参数含义参考info_gain()'''
        x, y = np.asarray(feature_x), np.asarray(y_labels)
        sample_weight = self._set_sample_weight(sample_weight)
        cond_ent = 0.
        for x_val in np.unique(x):
            x_idx = np.where(x == x_val)
            sub_x, sub_y = x[x_idx], y[x_idx]
            sub_samples = sample_weight[x_idx]
            p_i = 1.0 * len(sub_x) / len(x)
            cond_ent += p_i * self.cal_info_entropy(sub_y, sub_samples)
        return cond_ent

    def info_gain_rate(self, x, y, sample_weight=None):
        '''
        信息增益比
        :param x:
        :param y:
        :param sample_weight:
        :return:
        '''
        gain_rate = 1.0 * self.info_gain(x, y, sample_weight) / \
                    (1e-12 + self.cal_info_entropy(x, sample_weight))
        return gain_rate

    def info_gain(self, feature_x, y_labels, sample_weight):
        '''
        互信息/信息增益:H(y)-H(y|x)
        :param feature_x:
        :param y_labels:
        :param sample_weight:
        :return:
        '''
        gain=self.cal_info_entropy(y_labels,sample_weight)-\
            self.conditional_entropy(feature_x,y_labels,sample_weight)
        return gain



