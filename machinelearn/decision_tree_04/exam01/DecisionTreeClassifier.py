from machinelearn.decision_tree_04.exam01.Entropy_Utils import Entropy_Utils
from machinelearn.decision_tree_04.exam01.TreeNode import TreeNode
from machinelearn.decision_tree_04.exam01.DataBinWrapper import DataBinWrapper
import numpy as np


class DecisionTreeClassifier:
    '''
    决策树分类算法：包括ID3、C4.5和CART树，后剪技处理，针对连续特征数据，进行分箱处理
    '''

    def __init__(self, criterion='CART', is_feature_R=False, dbw_idx=None, \
                 max_depth=None, min_samples_split=2, min_samples_leaf=1, \
                 min_impurity_decrease=0, max_bins=10):
        self.utils = Entropy_Utils()  # 各种节点划分指标的计算
        self.criterion = criterion  # 结点划分标准，默认为CART
        if self.criterion.lower() == 'cart':  # 基尼指数，CART
            self.criterion_func = self.utils.gini_gain  #
        elif self.criterion.lower() == 'c45':  # 信息增益率， C4.5
            self.criterion_func = self.utils.info_gain_rate
        elif self.criterion.lower() == 'id3':  # 信息增益,ID3
            self.criterion_func = self.utils.info_gain
        else:
            raise ValueError('节点划分标准仅限CART,C4.5和ID3...')
        self.max_depth = max_depth  # 树的最大深度
        self.min_samples_split = min_samples_split  # 内部节点划分时的最小样本数
        self.min_samples_leaf = min_samples_leaf  # 叶子节点上的最小样本数
        self.min_impurity_decrease = min_impurity_decrease
        self.is_feature_R = is_feature_R  # 所有特征的数据是否都是连续实数
        self.dbw_idx = dbw_idx  # 需要进行分享的特征属性索引
        self.root_node: TreeNode() = None  # 根节点，类型为None
        self.dbw_XrangeMap = {}
        self.dbw = DataBinWrapper(max_bins=max_bins)  # 数据分箱处理，针对连续特征数据
        self.n_features = None  # 样本特征数
        self.prune_nums = 0  # 剪枝处理考察的节点数

    def _data_preprocess(self, x_samples):
        '''
        针对不同的数据类型做预处理，连续数据分享操作，否则不进行
        :param x_sample:
        :return:
        '''
        self.dbw_idx = np.asarray(self.dbw_idx)
        x_sample_prep = []  #
        if not self.dbw_XrangeMap:
            for i in range(x_samples.shape[1]):
                if i in self.dbw_idx:
                    self.dbw.fit(x_samples[:, i])
                    self.dbw_XrangeMap[i] = self.dbw.XrangeMap
                    x_sample_prep.append(self.dbw.transform(x_samples[:, i]))
                else:
                    x_sample_prep.append(x_samples[:, i])
        else:
            for i in range(x_samples.shape[1]):
                if i in self.dbw_idx:
                    x_sample_prep.append(self.dbw.transform(x_samples[:, i],
                                                            self.dbw_XrangeMap[i]))
                else:
                    x_sample_prep.append(x_samples[:, i])
        return np.asarray(x_sample_prep).T

    def fit(self, x_train, y_train, sample_weight=None):
        '''
        决策树的训练
        :param x_train:
        :param y_train:
        :param sample_weight:
        :return:
        '''
        print('决策树模型递归构建 ...')
        if y_train.dtype == 'object':
            raise ValueError('类别标签必须是数值型，请先编码..')
        n_samples, self.n_features = x_train.shape
        if sample_weight is None:
            sample_weight = np.asarray([1.0] * n_samples)
        if len(sample_weight) != n_samples:
            raise Exception('sample_weight size error:', len(sample_weight))
        self.root_node = TreeNode()  # 构建空的根节点
        if self.is_feature_R:  # 如果所有的数据都是连续
            self.dbw.fit(x_train)
            x_train = self.dbw.transform(x_train)
        elif self.dbw_idx is not None:
            x_train = self._data_preprocess(x_train)
