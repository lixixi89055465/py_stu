import torch

a=torch.rand(32,8)
b=torch.rand(32,8)
c=torch.stack([a,b],dim=0)
print(c.shape)


aa,bb=c.split([1,1],dim=0)
print(aa.shape, bb.shape)

aa,bb=c.split(1,dim=0)
print(aa.shape, bb.shape)

aa,bb=c.split([2,30],dim=1)
print(aa.shape, bb.shape)
aa,bb=c.split(1,dim=0)
print(aa.shape, bb.shape)
aa,bb=c.split(16,dim=1)
print(aa.shape, bb.shape)

b=torch.rand(32,8)
a=torch.rand(32,8)
c=torch.stack([a,b],dim=0)
print(c.shape)
aa,bb=c.chunk(2,dim=0)
print(aa.shape, bb.shape)
print(c.shape)
