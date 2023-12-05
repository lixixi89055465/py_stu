import pandas as pd
import numpy as np

c_data = pd.read_csv('../../data/watermelon.csv')
print(c_data.head())

feature_name = c_data.columns[:-1]
print(feature_name)
feature_split = pd.DataFrame()
for feat_name in feature_name:
    data_feat = c_data.loc[:, [feature_name, 'label']]
    data_sort=data_feat.sort_values
