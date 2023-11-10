import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.setdiff1d(a, b)
print(c)  # [1 2 3]

a = np.array([1, 2, 3])
b = np.array([1, 2, 3])
c = np.setdiff1d(a, b)
print(c)  # []

a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
c = np.setdiff1d(a, b)
print(c)  # [1]
a = np.array([1, 2, 3, 4])
b = np.array([3, 4, 5, 6])
c = np.setdiff1d(a, b)
print(c)  # [1 2]
a = np.array([1, 2, 3, 2, 4, 1])
b = np.array([3, 4, 5, 6])
c = np.setdiff1d(a, b)
print(c)  # [1 2]
a = np.array([8, 2, 3, 2, 4, 1])
b = np.array([7, 4, 5, 6, 3])
c = np.setdiff1d(a, b)
print(c)  # [1 2 8]
a = np.array([8, 2, 3, 2, 4, 1])
b = np.array([7, 4, 5, 6, 3])
c = np.setdiff1d(a, b)
print(c)  # [1 2 8]
a = np.array([8, 2, 3, 2, 4, 1])
b = np.array([7, 4, 5, 6, 3])
c = np.setdiff1d(a, b, True)
print(c)  # [8 2 2 1]
a = np.array([8, 2, 3, 4, 2, 4, 1])
b = np.array([7, 9, 5, 6, 3])
c = np.setdiff1d(a, b, True)
print(c)  # [8 2 4 2 4 1]