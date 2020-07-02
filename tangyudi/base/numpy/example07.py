import numpy as np

print(np.arange(15))
a = np.arange(15).reshape(3, 5)
print(a)

print(a.shape)
print(a.ndim)
print(a.dtype.name)
print(a.dtype)
print(a.size)
a = np.zeros((3, 4))
print(a)
a = np.ones((2, 3, 4), dtype=np.int32)
print(a)
a = np.arange(10, 30, 5)
print(a)
a = np.random.random((2, 3))
print(a)
from numpy import pi

a = np.linspace(0.2 * pi, 100)
print(a)
print(np.sin(300))
np.sin(np.linspace(0.2 * pi, 100))
print('1' * 100)
a = np.array([20, 30, 40, 60])
b = np.arange(4)
print(a)
print(b)
c = a - b
print(c)
print(c - 10)
A = np.array([
    [1, 1],
    [0, 1]]
)
B = np.array([
    [2, 0],
    [3, 4]])
print(A)
print(B)
print(A * B)
print(A.dot(B))
print(np.dot(A, B))
