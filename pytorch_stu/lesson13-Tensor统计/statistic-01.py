import torch

a = torch.arange(8).view(2, 4).float()
print(a)

print(a.min(), a.max(), a.mean(), a.prod())

print(a.sum())
print(a.argmax(), a.argmin())

a = torch.randn(4, 10)
print(a[0])

print(a.argmax())
print(a.argmax(dim=1))
