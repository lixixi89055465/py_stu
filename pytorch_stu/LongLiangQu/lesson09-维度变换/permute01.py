import torch

a = torch.rand(4, 3, 28, 28)
print(a.transpose(1, 3).shape)
b=torch.rand(4,3,28,32)
print(b.shape)
print(b.transpose(1, 3).shape)
print(b.transpose(1, 3).transpose(1, 2).shape)
print(b.permute(0, 2, 3, 1).shape)
