import torch

a = torch.randn(1, 1, 3)
print(a)
print(a.squeeze())
print(a.squeeze(0))
