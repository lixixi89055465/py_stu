'''

'''
import torch
from torch.nn import functional as F

# inputs = torch.randn(1, 2, 4, 4)
# print(inputs.size())
# print(inputs)
# unfold = torch.nn.Unfold(kernel_size=(2, 2), stride=2)
# patchs = unfold(inputs)
# print(patchs.size())
# print(patchs)
#
# print('1' * 100)
# fold = torch.nn.Fold(output_size=(4, 4), kernel_size=(2, 2), stride=2)
# inputs_restore = fold(patchs)
# print(inputs_restore.shape)
# print(inputs_restore)



x=torch.arange(0,1*3*4*5).float()
x=x.view(1,3,4,5)
print(x.shape)
print(x)
x1=F.unfold(x,kernel_size=3,dilation=1,stride=1)
print(x1.shape)
print(x1)
