import torch
import torch.nn as nn
from torch import autograd

m = nn.Conv3d(3, 3, (3, 7, 7), stride=1, padding=0)
input = autograd.Variable(torch.randn(1, 3, 7, 60, 40))
output = m(input)
print(output.size())
