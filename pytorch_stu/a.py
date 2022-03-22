'''

'''
import torch
a=torch.randn(7,512,5,5)
b,c,h,w=a.shape
print(a.shape)
b=a.permute(0,2,3,1).view(b,-1,c).contiguous()
print(b.shape)
