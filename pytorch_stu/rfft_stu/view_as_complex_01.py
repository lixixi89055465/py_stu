'''

'''
import torch
x=torch.randn(4,2)
print(x)
a=torch.view_as_complex(x)
print(a)
print('1'*100)
t=torch.rand(10,10)
rfft2=torch.fft.rfft2(t)
print(rfft2.size())
