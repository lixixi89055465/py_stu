import torch

w = torch.rand(16, 3, 5, 5)
b = torch.rand(16)

x = torch.rand(1, 3, 28, 28)
out = torch.nn.functional.conv2d(x, w, b, stride=1, padding=1)
print(out.shape)
out = torch.nn.functional.conv2d(x, w, b, stride=1, padding=1)
print(out.shape)
out=torch.nn.functional.conv2d(x,w,b,stride=2,padding=2)
print(out.shape)
