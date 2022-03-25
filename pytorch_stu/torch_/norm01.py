'''

'''
import torch
import torch.tensor as tensor

a=tensor([[1,2,3,4],
          [1,2,3,4]]).float()
print(a)
a0=torch.norm(a,p=2,dim=0)
a1=torch.norm(a,p=2,dim=1)
print(a0)
print(a1)
a0=torch.norm(a,p=0,dim=0)
a1=torch.norm(a,p=0,dim=1)
print(a0)
print(a1)

