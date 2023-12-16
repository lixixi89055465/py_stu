import numpy as np

x, y = np.ogrid[0:10:1, 0:10:2]
print(x, np.shape(x))
print('0' * 100)
print(y, np.shape(y))

x, y = np.ogrid[0:10:6j, 0:10:4j]
print('1'*100)
print(x, np.shape(x))
print(y, np.shape(y))

print('3'*100)
print(np.mgrid[0:10:2, 0:8:1])
print('4'*100)
print(np.ogrid[0:20:5, 0:8:1])