import torch

a = torch.randn(1, 2, 3)
print(a)
print(a.shape)
print(a[0])
print(a.shape)
print(a[0])
print(a[0, 1])
print(list(a.shape))
print(type(a.shape))
a = torch.randn([1, 3, 4])
print(a)
