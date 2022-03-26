import torch
a=torch.arange(8)-0.5
print(a)
b=torch.where(a>4)
print(b)
