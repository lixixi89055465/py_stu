import torch
import numpy as np
a=torch.tensor([[-1,2],[2,3]])
print(-a[:,0]+a[:,1])
b=a[:,0]+a[:,1]
print(b)
c=torch.Tensor([])
print(c)
print(torch.cat((c, b), dim=0))
