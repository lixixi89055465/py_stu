class TreeUtils:
    '''
    决策树工具类，封装书的信息
    '''

    def __init__(self, node_type, cls=None, best_feature=None):
        '''

        :param node_type:
        :param cls:
        :param best_feature:
        '''
        self.node_type = node_type  # 决策树节点类型（内部节点，叶子节点）
        # 键:最佳信息增益所对应的特征，值：以最佳特征的某个取值为根节点的子树
        self.tree_dict = dict()
        self.cls = cls  # 叶子节点所对应的列表值，若内部节点则值为None
        self.best_feature = best_feature

    def add_tree(self, key, tree):
        '''
        键：最佳信息增益所对应的特征，值：以最佳特征的某个取值为根节点的子树
        :param key:
        :param tree:
        :return:
        '''
        self.tree_dict[key] = tree

    def predict(self, test_sample):
        '''
        递归：根据构建的决策树进行预测
        :param test_sample:
        :return:
        '''
        if self.node_type == 'leaf' or (test_sample[self.best_feature] not in self.tree_dict):
            return self.cls
        tree = self.tree_dict.get(test_sample[self.best_feature])
        return tree.predict(test_sample)
