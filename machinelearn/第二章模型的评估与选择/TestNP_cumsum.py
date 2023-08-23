import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print('0' * 100)

print(np.cumsum(a))
print(np.cumsum(a, axis=0))
print(np.cumsum(a, axis=1))
