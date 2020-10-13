import numpy as np

a = np.array([[1, 2, 3], [7, 8, 9]])

b = np.array([[4, 5, 6], [1, 2, 3]])

print(a)

print(b)

c=np.c_[a,b]
print(c)
print('1'*100)
print(np.c_[a,b])