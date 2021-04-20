import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6], [11, 12, 13]])
special_col = np.arange(0, 3)
y = x + special_col
print(y)
