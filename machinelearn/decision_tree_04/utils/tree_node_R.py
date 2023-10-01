# -*- coding: utf-8 -*-
# @projectname  : py_stu
# @IDE:    : PyCharm
# @Time    : 2023/9/27 0:10
# @Author  : nanji
# @File    : tree_node_R.py
# @Description :
class TreeNode_R:
    '''
    决策树回归算法，书的节点信息封装,实体类，setXXX(),getXXX()
    '''

    def __init__(self, feature_idx: int = None, feature_val: int = None,
                 y_hat=None, square_error=None, n_samples: int = None,
                 criterion_val=None, left_child_Node=None, right_child_Node=None):
        '''
        决策树节点信息封装
        @param feature_idx: 特征索引，如果指定特征属性名称，可以按照索引取值
        @param feature_val: 特征取值
        @param square_error: 划分节点的标准：当前节点的平方误差
        @param n_samples: 当前节点所包含的样本量
        @param y_hat: 当前节点的预测值
        @param criterion_val:
        @param left_child_Node: 左子树
        @param right_child_Node: 右字数
        '''
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.square_error = square_error
        self.n_samples = n_samples
        self.y_hat = y_hat
        self.left_child_node = left_child_Node
        self.right_child_node = right_child_Node
        self.criterion_val=criterion_val
    def level_order(self):
        '''
        按层次遍历树
        @return:
        '''
        pass
