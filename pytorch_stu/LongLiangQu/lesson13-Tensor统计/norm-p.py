import torch

a = torch.full([8], 1,dtype=torch.float32)
print(a)
b=a.view(2,4)
print(b)
c=a.view(2,2,2)
print(b)

print('a.norm(1), b.norm(1), c.norm(1):')
print(a.norm(1), b.norm(1), c.norm(1))
print('a.norm(2), b.norm(2), c.norm(2):')
print(a.norm(2), b.norm(2), c.norm(2))
print('b.norm(1, dim=1):')
print(b.norm(1, dim=1))
print('b.norm(2, dim=1):')
print(b.norm(2, dim=1))
print('b.norm(2, dim=0):')
print(b.norm(2, dim=0))

print(c.norm(1, dim=0))
print(c.norm(2, dim=0))

