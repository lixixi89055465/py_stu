import numpy as np


class TreeNode(object):
    '''
    树节点实体类封装，略去get和setX方法，用于存储节点信息以及关联子节点，
    每个树节点包括内容：
    样本特征索引，样本特征取值，类别分布概率，权重分布概率，左孩子节点，
    右孩子节点，当前节点样本星
    以及当前最佳划分的节点标准（信息增益、信息增益率或基尼系数增益率）
    '''

    def __init__(self, feature_idx: int = None, feature_val=None, \
                 target_dist: dict = None, weight_dist: dict = None, \
                 left_child_node=None, right_child_node=None, \
                 n_samples: int = None, criterion_val: float = None):
        self.feature_idx = feature_idx  # 样本特征索引id
        self.feature_val = feature_val  # 样本特征的某个取值
        self.target_dist = target_dist
        self.weight_dist = weight_dist
        self.left_child_node = left_child_node
        self.right_child_node = right_child_node
        self.n_samples = n_samples
        self.criterion_val = criterion_val

    def level_order(self):
        pass

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
        sample_weight = self._set_sample_weight(sample_weight)
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


    def _set_sample_weight(self, sample_weight, n_sample):
        if sample_weight is None:
            sample_weight = [1.0] * n_sample
        return sample_weight
