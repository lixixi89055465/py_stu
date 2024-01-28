# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/9/27 0:10
# @Author  : nanji
# @File    : tree_node_R.py
# @Description :
class TreeNode_C:
    '''
    决策树分类算法，书的节点信息封装,实体类，setXXX(),getXXX()
    '''

    def __init__(self, feature_idx: int = None, feature_val: int = None,
                 criterion_val: float = None, n_samples: int = None,
                 target_dist: dict = None, weight_dist: dict = None,
                 left_child_Node=None, right_child_Node=None):
        '''
        决策树节点信息封装
        @param feature_idx: 特征索引，如果指定特征属性名称，可以按照索引取值
        @param feature_val: 特征取值
        @param criterion_val: 划分节点的标准：信息增益（率)，基尼指数增益
        @param n_samples: 当前节点所包含的样本量
        @param target_dist: 当前节点类别分布:0-25%,1-50%,2-25%
        @param weight_dist:  当前节点所包含的样本权重分布
        @param left_child_Node: 左子树
        @param right_child_Node: 右字数
        '''
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.criterion_val = criterion_val
        self.n_samples = n_samples
        self.target_dist = target_dist
        self.weight_dist = weight_dist
        self.left_child_node = left_child_Node
        self.right_child_node = right_child_Node

    def level_order(self):
        '''
        按层次遍历树
        @return:
        '''
        pass
