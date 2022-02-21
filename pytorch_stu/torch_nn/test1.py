'''
5.640.5.5
'''
from torch import nn
import torch
torch.randn
# x=torch.tensor([5,640,5,5],dtype=float)
x=torch.randn(5,640,5,5)
a=x.mean(1)
b=a.unsqueeze(1)
print(a.shape)
print(b.shape)
