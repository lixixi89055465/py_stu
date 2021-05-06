import numpy as np

h_forward = np.array([1, 2])
print(h_forward)
h_backward = np.array([3, 4])
print(h_backward)
h_bi = np.hstack((h_forward, h_backward))

print(h_bi)
