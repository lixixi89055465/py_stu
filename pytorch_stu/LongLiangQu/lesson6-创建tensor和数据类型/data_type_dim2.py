import torch

a = torch.ones(2)
print(a)
print(a.shape)
print(torch.Size([2]))

a = torch.randn(2, 3)
print(a.shape)
print(a.size(0))
print(a.size(1))
print(a.shape[1])
