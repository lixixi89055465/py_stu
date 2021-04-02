import torch
from torch import nn

print(torch.__version__)

w = torch.empty(3, 5)
print(nn.init.constant_(w, 0.3))