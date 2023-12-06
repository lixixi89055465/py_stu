import pandas as pd
import numpy as np

c_data = pd.read_csv('../../data/watermelon.csv').loc[:, ['密度', '含糖率', 'label']]
print(c_data.head())

feature_name = c_data.columns[:-1]
print(feature_name)
feature_split = pd.DataFrame()


def calc_entropy(sub_labels):
    '''计算信息墒 '''
    x_label_array = np.unique(sub_labels)
    entropy = 0.0
    for x_label in x_label_array:
        sub_y = sub_labels[sub_labels == x_label]
        p = len(sub_y) / len(sub_labels)
        entropy -= p * np.log2(p)  # 信息墒累加
    return entropy


for name in feature_name:
    data_feat = c_data[:, [name, 'label']]
    data_sort = data_feat.sort_values(by=name)
    feat_values = data_sort.iloc[:, 0].values
    split_t = (feat_values[1:] + feat_values[:-1]) / 2
    ent_gain = []
    entropy = calc_entropy(data_sort.iloc[:, 1])
    n_all = data_sort.shape[0]
    for i in range(len(split_t)):
        dt1 = data_sort[data_sort.iloc[:, 0] <= split_t[i]]
        dt2 = data_sort[data_sort.iloc[:, 0] > split_t[i]]
        ent1 = calc_entropy(dt1.iloc[:, -1])
        ent2 = calc_entropy(dt2.iloc[:, -1])
        entropy_sub = dt1.shape[0] / n_all * ent1 + dt2.shape[0] / n_all * ent2
        ent_gain.append(entropy - entropy_sub)
    feature_split[name + '划分点'] = split_t
    feature_split[name + '信息增益'] = ent_gain
    best_s=split_t[np.argmax(ent_gain)]
    max_gain=np.max(ent_gain)

