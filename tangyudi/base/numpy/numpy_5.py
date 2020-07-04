import numpy as np

data = np.sin(np.arange(20)).reshape(5, 4)
print(data)
ind = data.argmax(axis=0)
print(ind)
data_max = data[ind, range(data.shape[1])]
print(data_max)
print(data.shape[1])

a = np.arange(0, 40, 10)
print(a)
b = np.tile(a, (3, 5))
print(b)

a = np.array([[4, 3, 5], [1, 2, 1]])
print(a)
print('-' * 100)
b=np.sort(a, axis=0)
print(b)
