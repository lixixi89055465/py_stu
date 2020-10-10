import numpy as np
from scipy import linalg

a = np.array([[4, 12, -16],
              [12, 37, -43],
              [-16, -43, 98]])
L = linalg.cholesky(a, lower=True)
print(L)

print(np.allclose(np.dot(L, L.T), a))
