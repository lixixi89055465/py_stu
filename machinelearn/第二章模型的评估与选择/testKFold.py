import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [5, 9], [1, 5], [3, 9], [5, 8], [1, 1], [1, 4]])
y = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 0])
print('X:', X)
print('y:', y)
kf = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)
for train_index, test_index in kf.split(X, y):
    print('TRAIN:', train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print('0'*100)
