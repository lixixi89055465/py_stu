import torch
from torch import nn

w = torch.empty(3, 5)
nn.init.constant_(w, 0.3)
print(w)
