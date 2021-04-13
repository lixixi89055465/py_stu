import torch

b = torch.rand([1, 32, 1, 1])
print(b.shape)
print(b.repeat(4, 32, 1, 1).shape)
b1 = b.repeat(4, 32, 1, 1)
print(b[0, 2, 0, 0])
print(b1[0, 2, 0, 0])
print(b1[1, 2, 0, 0])
print(b1[2, 2, 0, 0])
print(b1[3, 2, 0, 0])

print(b.repeat(4, 1, 1, 1).shape)
print(b.repeat(4, 1, 32, 32).shape)
a=torch.randn(3,4)
print(a.t())
print(a.T)
