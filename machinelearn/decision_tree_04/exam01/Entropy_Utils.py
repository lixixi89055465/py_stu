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
            ent_x += -p_i * np.log2(p_i)
        return ent_x

    def _set_sample_weight(self, sample_weight, n_samples):
        if sample_weight is None:
            sample_weight = np.asarray([1.0] * n_samples)
        return sample_weight

    def conditional_entropy(self, feature_x, y_labels, sample_weight):
        '''计算条件熵 H(y|x),参数含义参考info_gain()'''
        x, y = np.asarray(feature_x), np.asarray(y_labels)
        sample_weight = self._set_sample_weight(sample_weight,len(y_labels))
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

    def info_gain(self, feature_x, y_labels, sample_weight=None):
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
    def cal_gini(self, y, sample_weight=None):
        '''
        计算基尼系数
        :param y:
        :param sample_weight:
        :return:
        '''
        y = np.asarray(y)
        sample_weight = self._set_sample_weight(sample_weight, len(y))
        y_values = np.unique(y)
        gini = 1.0
        for val in y_values:
            p_i = 1.0 * len(y[val == y]) * np.mean(sample_weight[y == val]) / len(y)
            gini -= p_i * p_i
        return gini

    def conditional_gini(self, feature_x, y_labels, sample_weight=None):
        '''
        计算条件gini系数 ：Gini(y|x)
        :param feature_x:
        :param y_labels:
        :param sample_weight:
        :return:
        '''
        x, y = np.asarray(feature_x), np.asarray(y_labels)
        sample_weight = self._set_sample_weight(sample_weight,len(y_labels))
        cond_gini = .0  # 计算条件 gini系数
        for x_val in np.unique(x):
            x_idx = np.where(x_val == x)
            sub_x, sub_y, sub_sample_weight = x[x_idx], y[x_idx], sample_weight[x_idx]
            p_i = 1.0 * len(x_idx) / len(x)
            cond_gini += p_i * self.cal_gini(sub_y, sub_sample_weight)
        return cond_gini

    def gini_gain(self, feature_x, y_labels, sample_weight=None):
        '''
        计算gini值的增益Gini(D) -Gini(y|x)
        :param feature_x:
        :param y_labels:
        :param sample_weight:
        :return:
        '''
        g_gain = self.cal_gini(y_labels, sample_weight) - \
                 self.conditional_gini(feature_x, y_labels, sample_weight)
        return g_gain

    def square_error(self, y, sample_weight):
        y = np.asarray(y)
        return np.sum((y - np.mean(y)) ** 2 * sample_weight)

    def cond_square_error(self, x, y, sample_weight):
        x, y = np.asarray(x), np.asarray(y)
        error = .0
        for x_v in set(x):
            x_index = np.where(x_v == x)
            new_y = y[x_index]
            new_sample = y[x_index]
            error += self.square_error(new_y, new_sample)
        return error

    def square_error_gain(self, x, y, sample_weight=None):
        '''

		:param x:
		:param y:
		:param sample_weight:
		:return:
		'''
        sample_weight = self._set_sample_weight(sample_weight, len(x))
        return self.square_error(y, sample_weight) - self.cond_square_error(x, y, sample_weight)

    def _set_sample_weight(self, sample_weight, x_num):
        if sample_weight is None:
            self.sample_weight = np.ones((x_num))
        return self.sample_weight



