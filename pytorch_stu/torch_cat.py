'''

'''

import torch

x=torch.ones([2,3])
print(x.shape)
y=torch.cat([x,x,x*0.5],0)
print(y.shape)
print(y)
print(x * 0.5)

