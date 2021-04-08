import torch

B = torch.tensor([[1, 2, 9], [2, 0, 4], [3, 4, 5]])
print(B)

B = B.T
print(B)

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A)
B = A.clone()
print(A, A + B)
print(id(A), id(B))

x = torch.arange(4, dtype=torch.float32)
print(x, x.sum())
A = torch.arange(20 * 2).reshape(2, 5, 4)
print(A.shape, A.sum())
