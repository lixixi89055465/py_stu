'''

'''

import torch
from torch.nn import functional as f

x = torch.arange(0, 1 * 3 * 15 * 15).float()
x = x.view(1, 3, 15, 15)
print(x)
x1 = f.unfold(x, kernel_size=3, dilation=1, stride=1)
print(x1.shape)
B, C_kh_kw, L = x1.size()
x1 = x1.permute(0, 2, 1)
x1 = x1.view(B, L, -1, 3, 3)
print(x1)
