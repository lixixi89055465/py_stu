import torch

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))

print(a + b)
# 遇到1可以进行广播
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((2, 1))
print(a + b)
