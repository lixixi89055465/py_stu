import torch
import numpy as np

a = torch.tensor([1.1])
print(a)
b = torch.tensor([1.1, 2.2])
print(b)
print(torch.FloatTensor(1))

print(torch.FloatTensor(1))

data=np.ones(2)

print(data)

print(torch.from_numpy(data))

data=np.ones(2)
print(data)
print(torch.from_numpy(data))
a=torch.ones(2)
print(a.shape)
print(torch.Size([2]))