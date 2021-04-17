import torch

x = torch.rand(1, 16, 7, 7)
out = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
print(out.shape)
out = torch.nn.functional.interpolate(x, scale_factor=3, mode='nearest')
print(out.shape)
