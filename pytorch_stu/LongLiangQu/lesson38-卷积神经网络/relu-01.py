import torch

x = torch.rand(1, 16, 7, 7) * 2 - 1
print(x.min())
layer = torch.nn.ReLU(inplace=True)
out = layer.forward(x)
print(out.min())
print(out.shape)
out = layer(x)
print(out.shape)
print(out.min())

out = torch.nn.functional.relu(x)
print(out.shape)
print(out.min())
print(x.min())

layer = torch.nn.LeakyReLU(inplace=True)
out = layer.forward(x)
print(out.min())
print(out.shape)
out = layer(x)
print(out.shape)
print(out.min())
