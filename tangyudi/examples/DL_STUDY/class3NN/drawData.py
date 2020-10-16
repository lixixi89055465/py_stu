import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(10)

N = 100  # number of points per class
D = 2  # dimensionality
K = 3
X = np.zeros((N * K, D))

y = np.zeros(N * K, dtype='uint8')

# for j in range(K):
#     ix = range(N * j, N * (j + 1))
#     r = np.linspace(0.0, 1, N)
#     # t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
#     t = np.linspace(j * 4, (j + 1) * 4, N)
#     X[ix] = np.c_[r * np.sin(t)*2, r * np.cos(t)*2]
#     y[ix] = j
#


for j in range(K):
    ix=range(N*j,N*(j+1))
    r=np.linspace(0.0,1,N)
    t=np.linspace(j*4,(j+1)*4,N)+np.random.randn(N)*0.2
    X[ix]=np.c_[r*np.sin(t)*2,r*np.cos(t)*2]
    y[ix]=j

fig = plt.figure()

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.show()
