import torch
a=torch.full([4],1/4.)
print(a)
print(a * torch.log2(a))
print(-(a * torch.log2(a)).sum())

a=torch.tensor([0.1,0.1,0.1,0.7])
print(-(a * torch.log2(a)).sum())
a=torch.tensor([0.001,0.001,0.001,0.999])
print(-(a * torch.log2(a)).sum())


