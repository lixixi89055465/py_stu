'''

'''
import torch
import torch.nn.functional as F
import numpy as np
filters = torch.randn(8,4,3,3)
inputs = torch.randn(1,4,5,5)
a=F.conv2d(inputs, filters, padding=1)
print(a.shape)
