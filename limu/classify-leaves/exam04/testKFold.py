import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [5, 9], [1, 5], [3, 9], [5, 8], [1, 1], [1, 4]])
y = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 0])

print('X:', X)
print('y:', y)

kf = KFold(n_splits=2, shuffle=True, random_state=2020)
for train_indx, test_indx in kf.split(X):
	print('2' * 100)
	print(train_indx)
	print(test_indx)
print('1' * 100)

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [5, 9], [1, 5], [3, 9], [5, 8], [1, 1], [1, 4]])
y = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 0])

print('X:', X)
print('y:', y)
print('2' * 100)
skf = StratifiedKFold(n_splits=2, random_state=2020, shuffle=True)
print(skf)
# 做划分是需要同时传入数据集和标签
for train_idx, test_idx in skf.split(X, y):
	print('Train:', train_idx, "\t Test:", test_idx)
