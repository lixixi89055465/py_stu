class DecisionTreeClassifier:
    '''
    决策树分类算法：包括ID3、C4.5和CART树，后剪技处理，针对连续特征数据，进行分箱处理
    '''

    def __init__(self, criterion='CART', is_feature_R=False, dbw_idx=None, \
                 max_depth=None, min_samples_split=2, min_samples_leaf=1, \
                 min_impurity_decrease=0, max_bins=10):
        pass
