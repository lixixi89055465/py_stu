import torch

print(torch.empty(2, 3))
torch.empty((2,3), dtype=torch.int64)

print('0'*100)
a = torch.empty_strided((2, 3), (1, 2))
print(a)
print(a.stride())