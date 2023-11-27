import numpy as np
from machinelearn.decision_tree_04.exam01.TreeUtils import TreeUtils
from machinelearn.decision_tree_04.exam01.Entropy_Utils import Entropy_Utils


class ID3_DecisionTree:
    '''
    信息增益方法创建决策树，简单实现，无剪枝，仅包含划分节点的最小信息增益标准
    '''
    LEAF = 'leaf'  # 叶子节点类型
    INTERNAL = 'internal'  # 内部节点类型

    def __init__(self, min_info_gain, feature_names):
        self.min_info_gain = min_info_gain  # 最小信息增益阈值
        self.class_nums = None  # 类别数目
        self.feature_names = feature_names  # 特征属性名称
        self.cal_info_gain = Entropy_Utils.info_gain()  # 计算信息增益函数
        self.y_labels_dict = {}  # 样本类别字典

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
        if len(label_unique) == 1:
            print('类别', label_unique[0])
            return TreeUtils(self.LEAF, cls=label_unique[0])
        class_lab_nums = []
        for lab in label_unique:
            class_lab_nums.append([lab, len(y_train[y_train == lab])])
        # 各类别所包含的样本量中的最大值所对应的类别索引
        class_idx = max(class_lab_nums, key=lambda x: x[1])
        # 特征集为空，投票法，取样本呢量最大所对应的类被
        if len(feature_idx) == 0:  # 叶子节点
            print('类别:', self.y_labels_dict[class_idx])
            return TreeUtils(self.LEAF, cls=class_idx)

        # 计算每个特征的信息增益，并选择信息增益最大的特征作为划分节点依据
        best_feature_idx, best_gain = 0, 0
        for idx in feature_idx:  # 针对每个特征计算信息增益
            feature_data = x_train[:, idx]
            feature_gain = self.cal_info_gain(feature_data, y_train)  # 信息增益
            print(f'特征:{self.feature_names[idx]},信息增益是：{feature_gain}')
            if feature_gain > best_gain:
                best_feature_idx, best_gain = idx, feature_gain
        # 信息增益不足以划分节点 ，标记为叶子界定啊
        if best_gain < self.min_info_gain:  # 叶子节点
            print('类别;', self.y_labels_dict[class_idx])
            return TreeUtils(self.LEAF, cls=class_idx)
        tree = TreeUtils(self.INTERNAL, best_feature=best_feature_idx)  # 构造决策树内部节点
        print('-' * 100)
        print(f'最佳划分特征{self.feature_names},最大信息增益 : {best_gain}')
        print('=' * 100)
        # 递归时，出去已经被选择的特征属性，如“纹理”
        sub_feature_idx = list(filter(lambda x: x != best_feature_idx, feature_idx))
        # 当前最佳划分节点的不同样本值，如【清晰，模糊，烧糊】
        best_feature_unique_value = np.unique(x_train[:, best_feature_idx])
        # 以不同的样本值划分数据集，递归创建树
        for f_val in best_feature_unique_value:
            print(f'当特征{self.feature_names[best_feature_idx]},值为{f_val}时 ')
            sub_train_set = x_train[x_train[:, best_feature_idx] == f_val]
            sub_train_label = y_train[x_train[:, best_feature_idx == f_val]]
            sub_tree = self._build_tree(sub_train_set, sub_train_label, sub_feature_idx)
            tree.add_tree(f_val, sub_tree)
        return tree

    def predict(self, x_test, y_test, tree: TreeUtils):
        '''
        循环对每个测试样本进行预测
        :param x_test:
        :param y_test:
        :param tree:
        :return:
        '''
        values = list(self.y_labels_dict.values())
        y_test_pred, y_test_list = [], []  # 预测结果列表和测试样本编码
        for i, x in enumerate(x_test):
            y_test_list.append(values.index(y_test[i]))
            label = tree.predict(x)  # 可能为none
            if label in values:
                y_test_pred.append(values.index(label))
            else:
                y_test_pred.append(len(values))  # 可能为none
        return np.asarray(y_test_list), np.array(y_test_pred)
