'''

'''

import torch
import torch.nn as nn



inp = torch.tensor([[[[1.0, 2, 3, 4, 5, 6],
                      [7, 8, 9, 10, 11, 12],
                      [13, 14, 15, 16, 17, 18],
                      [19, 20, 21, 22, 23, 24],
                      [25, 26, 27, 28, 29, 30],
                      [125, 126, 217, 218, 219, 310],
                      ]]])

print('inp=')
print(inp)
unfold=nn.Unfold(kernel_size=(3,3),dilation=1,padding=0,stride=(3,3))
inp_unf=unfold(inp)

print('inp_unf=')
print(inp_unf)
print(inp_unf.shape)
