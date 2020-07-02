
import numpy as np
a=np.arange(12)
b=a
print(b is a)
b.shape=3,4
print(a.shape)
print(a)
print(b)
print(id(a))
print(id(b))
c=a.view()
print(c)
a.shape=2,6
print(a)
a[0,4]=55
print(a)
print(c)

print(c is a)
d=a.copy()
print(d is a)
d[0,0]=9999
print(d)
print(a)
