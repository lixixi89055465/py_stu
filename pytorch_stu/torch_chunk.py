import torch
import torch.nn as nn

a = torch.Tensor([[1, 2, 4]])
b = torch.Tensor([[4, 5, 7], [3, 9, 8], [9, 6, 7]])
c = torch.cat((a, b), dim=0)
print(c.shape)
print(c.size())
print(type(c))
print('#' * 100)
d = torch.chunk(c, 1, dim=0)
print(d)
print(len(d))
print(type(d))