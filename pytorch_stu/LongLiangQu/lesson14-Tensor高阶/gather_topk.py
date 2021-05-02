import torch

cond = torch.randn(2, 2)
print(cond)

a = torch.zeros(2, 2)
print(a)
b = torch.ones(2, 2)
print(b)
c = torch.where(cond > 0.5, a, b)
print(c)

prob=torch.randn(4,10)
idx=prob.topk(dim=1,k=3)
print(idx)
idx=idx[1]
print(idx)
label=torch.arange(10)+100
print(label.expand(4, 10))
torch.gather(label.expand(4,10),dim=1,index=idx.long())