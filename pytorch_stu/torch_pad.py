'''

'''


import torch
import torch.nn.functional as F
original_values = torch.randn([2,1, 3, 2])

padding_values=F.pad(original_values,pad=(1,1,2,2,1,1),mode='constant',value=0)
print(padding_values)
print(padding_values.shape)