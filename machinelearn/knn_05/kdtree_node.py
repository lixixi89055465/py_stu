# -*- coding: utf-8 -*-
# @Time    : 2023/10/2 14:46
# @Author  : nanji
# @Site    : 
# @File    : kdtree_node.py
# @Software: PyCharm 
# @Comment : 3. k近邻算法实现——𝒌𝒅树
class KDTreeNode:
    '''
    KD树结点信息封装
    '''

    def __init__(self, instance_node=None, instance_label=None, instance_idx=None,
                 split_feature=None, left_child=None, right_child=None, kdt_depth=None):
        '''
        用于封装kd树的结点信息结构
        :param instance_node: 实例点，一个样本
        :param instance_label: 实例点对应的类别标记
        :param instance_idx: 该实例点对应的样本索引，用于kd树的可视化
        :param split_feature: 划分的特征属性,x^(i)
        :param left_child: 左子树，小于切分点的
        :param right_child: 右子树，大于切分点
        :param kdt_depth: kd树的深度
        '''
        self.instance_node = instance_node
        self.instance_label = instance_label
        self.instance_idx = instance_idx
        self.split_feature = split_feature
        self.left_child = left_child
        self.right_child = right_child
        self.kdt_depth = kdt_depth
