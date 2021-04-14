import torch

cond = torch.randn(2, 2)
print(cond)

a = torch.zeros(2, 2)
print(a)
b = torch.ones(2, 2)
print(b)
c = torch.where(cond > 0.5, a, b)
print(c)
