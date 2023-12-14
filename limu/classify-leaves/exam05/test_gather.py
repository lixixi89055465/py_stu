import torch

a = torch.Tensor([[1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5]])
print(a)
b = a.gather(1, torch.tensor([[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]]))
print(b)
