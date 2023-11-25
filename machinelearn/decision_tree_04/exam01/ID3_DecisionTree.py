import numpy as np
from machinelearn.decision_tree_04.exam01.TreeUtils import TreeUtils
from machinelearn.decision_tree_04.exam01.Entropy_Utils import Entropy_Utils


class ID3_DecisionTree:
    '''
    信息增益方法创建决策树，简单实现，无剪枝，仅包含划分节点的最小信息增益标准
    '''
    LEAF = 'leaf'  # 叶子节点类型
    INSTERNAL = 'internal'  # 内部节点类型

    def __init__(self, min_info_gain, feature_names):
        self.min_info_gain = min_info_gain  # 最小信息增益阈值
        self.class_nums = None  # 类别数目

    def fit(self, x_train, y_train):
        '''
        训练模型，生产决策树，首先对类别标签进行处理，然后递归调用
        :param x_train:
        :param y_train:
        :return:
        '''
        # 类别标签编码
        for i, label in enumerate(list(set(y_train))):
            self.y_labels_dict[i] = label  # 以label为值，相应位置索引为键
        feature_idx = list(range(x_train.shape[1]))  # 特征索引列表
        tree = self._build_tree(x_train, y_train, feature_idx)
        return tree

    def _build_tree(self, x_train, y_train, feature_idx):
        '''
        递归实现ID3算法，以特征最大的信息增益为划分节点的标准
        :param x_train:
        :param y_train:
        :param feature_idx:
        :return:
        '''
        label_unique = np.unique(y_train)  # 当前样本类别不同的取值
        if len(label_unique) == 1:  # 当前节点包含的样本全属于同一类别，无需划分
            print('类别', label_unique[0])
            return TreeUtils(self.LEAF, cls=label_unique[0])
        class_lab_nums = []  # 各样本类别数，如（是、）（否、）=(0,9),(1,8 )
        for lab in label_unique:
            class_lab_nums.append([lab, len(y_train[y_train == lab])])
        # 各类别所包含的样本量中的最大值所对应的类别索引
        class_idx = max(class_lab_nums, key=lambda x: x[1])
