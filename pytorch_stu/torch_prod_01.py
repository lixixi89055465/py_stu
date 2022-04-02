import torch
a=torch.randn(1,3)
print(a)
b=torch.prod(a)
print(b)
a=torch.randn(4,2)
print(a)
b=torch.prod(a,1)
print(b)