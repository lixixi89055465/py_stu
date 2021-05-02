import torch

a = torch.rand(2, 3, 28, 28)
# print(a)
print(a.shape)

print('-'*30)
print(3*2*28*28)
print(a.numel())
print(a.dim())
a=torch.tensor(1)
print(a)
print(a.dim())