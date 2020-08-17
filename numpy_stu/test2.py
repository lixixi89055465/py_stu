import numpy as np
from scipy import linalg

a = np.array([[2, 4, -2],
              [1, -3, -3],
              [4, 2, 2]])
b = np.array([2, -1, 3])
x = linalg.solve(a, b)
print(x)
print(np.dot(a, x))
print(b)
