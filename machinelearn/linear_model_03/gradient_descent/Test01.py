
import numpy as np
a=np.random.rand(2,5)
print(a)
print(a.mean(axis=1))
print(a.mean(axis=1,keepdims=False))