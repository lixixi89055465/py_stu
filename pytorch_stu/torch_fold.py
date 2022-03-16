'''

'''
import torch
from torch.nn import functional as F

# w=torch.rand(16,3,5,5)
# b=torch.rand(16)
# x=torch.randn(1,3,28,28)
# out=F.conv2d(x,w,b,stride=1,padding=1 )
# print(out.shape)
# out=F.conv2d(x,w,b,stride=2,padding=1)
# print(out.shape)
# print('1'*30)
# out=F.conv2d(x,w)
# print(out.shape)


inputs = torch.randn(1, 2, 4, 4)
print(inputs.size())
print(inputs)
unfold = torch.nn.Unfold(kernel_size=(2, 2), stride=2)
patchs = unfold(inputs)
print(patchs.size())

print('1' * 100)
fold = torch.nn.Fold(output_size=(4, 4), kernel_size=(2, 2), stride=2)
inputs_restore = fold(patchs)
print(inputs_restore.shape)
print(inputs_restore)
