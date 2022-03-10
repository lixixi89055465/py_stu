'''

'''

import torch
import torch.nn.functional as F
original_values=torch.randn([2,1,3,2])
print('original_values:',original_values)
print('original_values 的 shape',original_values.shape )
padding_values=F.pad(original_values,pad=(1,0,0,0,0,0),mode='constant',values=0)
print('padding_values:',padding_values)
print('padding_values.shape',padding_values.shape)
