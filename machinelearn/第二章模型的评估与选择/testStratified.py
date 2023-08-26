import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [5, 9], [1, 5], [3, 9], [5, 8], [1, 1], [1, 4]])
y = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 0])
print(X)
print(y)
skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
print(skf)
print('0' * 100)
for train_index, test_index in skf.split(X, y):
    print(train_index)
    break
