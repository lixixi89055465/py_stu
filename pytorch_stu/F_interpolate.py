'''

'''

import torch

x = torch.autograd.Variable(torch.randn([1, 3, 64, 64]))
y0 = torch.nn.functional.interpolate(x, scale_factor=0.5)
print(y0.shape)
