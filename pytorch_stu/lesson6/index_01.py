import torch

a = torch.rand(4, 3, 28, 28)
print(a[0].shape)
print(a[0, 0].shape)
print(a[0, 0, 24, 24])
print(a.shape)
print(a[:2].shape)
print(a[:2, :1, :, :].shape)
print(a[2:, :1, :, :].shape)
print(a[2:, -1:, :, :].shape)
print(a[:, :, 0:28:2, 0:28:3].shape)
print(a[:, :, ::2, ::2].shape)
print(a.shape)
