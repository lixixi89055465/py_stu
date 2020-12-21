import numpy as np
import pandas as pd

columns = pd.read_table("diabetes_data_upload.csv", delimiter=',')

for col in columns.columns:
    if col != 'Age' and col != 'Gender' and col != 'class':
        columns[col] = [1 if c == 'Yes' else 0 for c in columns[col]]
    if col == 'class':
        columns[col] = [1 if c == 'Positive' else 0 for c in columns[col]]
    if col == 'Gender':
        columns[col] = [1 if c == 'Male' else 0 for c in columns[col]]

from sklearn import tree

dtr = tree.DecisionTreeRegressor(max_depth=10)

# dtr.fit(housing.data[:, [6, 7]], housing.target)
# dtr.fit()
dtr.fit(columns.values[:, :15], columns.values[:, 16])
dot_data = \
    tree.export_graphviz(
        dtr,
        out_file=None,
        feature_names=columns.columns,
        filled=True,
        impurity=False,
        rounded=True
    )
import pydotplus

graph = pydotplus.graph_from_dot_data(dot_data)
graph.get_nodes()[7].set_fillcolor("#FFF2DD")
from IPython.display import Image

Image(graph.create_png())

graph.write_png("dtr_white_background.png")

from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(columns.values[:, :15], columns.values[:, 16],
                                                                    test_size=0.1, random_state=43)
dtr = tree.DecisionTreeRegressor(max_depth=10)

dtr.fit(data_train, target_train)
