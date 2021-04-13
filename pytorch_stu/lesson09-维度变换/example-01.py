import torch
a = torch.rand(4, 1, 28, 28)
print(a.shape)
print(a.view(4, 28, 28).shape)

print(a.view(4, 28 * 28).shape)
print(a.view(4 * 28, 28).shape)
print(a.view(4 * 1, 28, 28).shape)
b=a.view(4,784)
print(b.view(4, 28, 28, 1, 1).shape)
