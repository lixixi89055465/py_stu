import numpy as np

weight = 2 * np.random.randn(2, 3) - 1
print(weight)
weight[0][0] = 0
sign = np.zeros_like(weight)
sign = np.where(weight > 0, 1, weight)
sign = np.where(weight < 0, -1, sign)
sign = np.where(weight == 0, 0, sign)
print(sign)
