from machinelearn.decision_tree_04.exam01.Entropy_Utils import Entropy_Utils
from machinelearn.decision_tree_04.exam01.TreeNode import TreeNode
from machinelearn.decision_tree_04.exam01.DataBinWrapper import DataBinWrapper
import numpy as np
import time


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
        x_sample_pred = []
        if not self.dbw_XrangeMap:
            for i in range(x_samples.shape[0]):
                if i in self.dbw_idx:
                    print('对特征索引为%d 的样本进行分享处理' % i)
                    self.dbw.fit(x_samples[:, i])
                    print('分箱点为：', self.dbw.XrangeMap)
                    print('-' * 100)
                    self.dbw_XrangeMap[i] = self.dbw.XrangeMap
                    x_sample_pred.append(self.dbw.fit(x_samples[:, i]))
                else:
                    x_sample_pred.append(x_samples[:, i])
        else:
            for i in range(x_samples.shape[1]):
                if i in self.dbw_idx:
                    x_sample_pred.append(self.dbw.transform(x_samples[:, i], \
                                                            self.dbw_XrangeMap[i]))
                else:
                    x_sample_pred.append(x_samples[:, i])
        return np.asarray(x_sample_pred).T

    def fit(self, x_train, y_train, sample_weight=None):
        '''
        决策树的训练
        :param x_train:
        :param y_train:
        :param sample_weight:
        :return:
        '''
        print('决策树模型递归构建 starting...')
        if y_train.dtype == 'object':
            raise ValueError('类别标签必须是数值型，请先编码...')
        n_samples, self.n_features = x_train.shape
        if not sample_weight:
            sample_weight = np.asarray([1.0] * n_samples)
        if len(sample_weight) != n_samples:
            raise Exception('sample weight size error:', len(sample_weight))
        self.root_node = TreeNode()  # 构建空的根节点
        if self.is_feature_R:
            self.dbw.fit(x_train)
            x_train = self.dbw.transform(x_train)
        elif self.dbw_idx:
            x_train = self._data_preprocess(x_train)
        # 递归时间
        time_start = time.time()
        self._build_tree(1, self.root_node, x_train, y_train, sample_weight)
        time_end = time.time()
        print('决策树模型递归构建完成，耗时：%f second' % (time_end - time_start))

    def _build_tree(self, cur_depth, cur_node: TreeNode, x_train, y_train, sample_weight):
        '''
        递归进行特征选择 ,构建树
        :param param:
        :param root_node:
        :param x_train:
        :param y_train:
        :param sample_weight:
        :return:
        '''
        n_samples, n_features = x_train.shape
        target_dist, weight_dist = {}, {}
        class_labels = np.unique(y_train)
        for label in class_labels:
            target_dist[label] = len(y_train[y_train == label]) / n_samples
            weight_dist[label] = np.mean(weight_dist[y_train == label])
        cur_node.target_dist = target_dist  # 类别分布
        cur_node.weight_dist = weight_dist  # 权重分布
        cur_node.n_samples = n_samples  # 样本量
        if len(target_dist) <= 1:
            return
        if n_samples < self.min_samples_split:
            return
        if self.max_depth is not None and cur_depth > self.max_depth:
            return
            # 寻找最佳的特征以及取值
        best_idx, best_index_val, best_criterion_val = None, None, 0

    def predict(self, x_test):
        '''
        预测样本 所属类别，预测类别概率为二维数组
        :param x_test:
        :return:
        '''
        return np.argmax(self.predict_probability(x_test), axis=1)

    def predict_probability(self, x_test, root_node=None):
        '''
        预测测试样本的概率分布
        :param x_test:
        :param root_node:
        :return:
        '''
        if self.is_feature_R:
            x_test = self.dbw.transform(x_test)
        elif self.dbw_idx is not None:
            x_test = self._data_preprocess(x_test)
        time_start = time.time()
        prob_dist = []
        class_num = len(self.root_node.target_dist)
        for i in range(x_test.shape[0]):
            prob_dist.append(self._search_node(self.root_node, x_test[i], class_num))
        time_end = time.time()
        print('对测试样本进行预测（即从根节点到叶子节点搜索），耗时:%f second' % \
              (time_end - time_start))
        return np.asarray(prob_dist)

    def _search_node(self, cur_node, x_test, class_num):
        '''
        检索叶子节点的结果，用于预测，即在根节点到叶子节点的一条路径上搜索
        :param cur_node:  当前决策树根节点
        :param x_test:  每个测试样本
        :param class_num:  类别数
        :return:
        '''
        if cur_node.left_child_node and x_test[cur_node.feature_idx] == cur_node.feature_val:
            return self._search_node(cur_node.left_child_node, x_test, class_num)
        elif cur_node.right_child_node and x_test[cur_node.feature_idx] != cur_node.feature_val:
            return self._search_node(cur_node.right_child_node, x_test, class_num)
        else:
            class_p = np.zeros(class_num)
            for c in range(class_num):
                class_p[c] = cur_node.target_dist.get(c, 0) * cur_node.weight_dist.get(c, 1.0)
            class_p /= np.sum(class_p)
            return class_p
