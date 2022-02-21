'''

'''

import torch
import torch.nn as nn

inputs = torch.Tensor(5, 4, 512, 512)
print(inputs.shape)
unfold = nn.Unfold(kernel_size=4, dilation=1, padding=0, stride=4)
out=unfold(inputs)
print(out.shape)
