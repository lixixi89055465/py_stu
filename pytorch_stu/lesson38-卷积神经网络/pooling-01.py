import torch

x = torch.rand(1, 16, 14, 14)
layer = torch.nn.MaxPool2d(2, stride=2)

out = layer(x)
print(out.shape)
out = torch.nn.functional.avg_pool2d(x, 2, stride=2)
print(out.shape)

layer = torch.nn.MaxPool2d(2, stride=2)
out = layer(x)
print(out.shape)
out = torch.nn.functional.avg_pool2d(x, 2, stride=2)
print(out.shape)
