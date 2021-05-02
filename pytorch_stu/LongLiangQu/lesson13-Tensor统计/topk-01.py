import torch

a = torch.rand(4, 10) * 2 - 1
print(a.shape)
print(a)
print('a.topk(3, dim=1):')
print(a.topk(3, dim=1))
print(a.topk(3, dim=1, largest=False))

print('-' * 20)
print(a.kthvalue(8, dim=1))
print('a:')
print(a)
print('-=' * 20)
print('a.kthvalue(3):')
print(a.kthvalue(3))
