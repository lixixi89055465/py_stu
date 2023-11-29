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
