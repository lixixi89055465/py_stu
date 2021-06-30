import torch
from torch import nn
import torch.nn.functional as F

input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
print(input)
x=F.interpolate(input,scale_factor=2,mode='nearest')
print(x)
x=F.interpolate(input,scale_factor=2,mode='bilinear',align_corners=True)
print(x)
