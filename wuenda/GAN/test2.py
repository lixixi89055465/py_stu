import torch
from torch import nn

print(torch.__version__)

w = torch.empty(3, 5)
print("w:")
print(torch.max(w))
print(w)
a = nn.init.normal_(w)
print(a)
print('+'*20)
print(torch.empty(2, 3))