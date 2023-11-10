
import numpy as np
a=np.random.rand(2,5)
print(a)
print('0'*100)
print(a.mean(axis=1, keepdims=False))
b=np.random.random(size=(2,5))
print(b)

c=np.random.randn(2,5)
print(c)