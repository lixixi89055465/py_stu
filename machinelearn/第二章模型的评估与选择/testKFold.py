import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [5, 9], [1, 5], [3, 9], [5, 8], [1, 1], [1, 4]])
y = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 0])
print('X:', X)
print('y:', y)
kf = KFold(n_splits=2, random_state=2020, shuffle=True)
print(kf)
print('0' * 100)
for train_index, test_index in kf.split(X):
    print('Train:', train_index, 'Test:', test_index)
    X_train, y_trian = X[train_index], y[test_index]
    y_train, y_test = y_train[train_index], y_test[test_index]
    
