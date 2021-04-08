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

sum_A = A.sum(axis=1, keepdims=True)
print(sum_A)

print(A / sum_A)

print('*' * 20)
print(A)
A_sum_axis0 = A.sum(axis=0)
print('A_sum_axis0:')
print(A_sum_axis0)

print(A.sum() / A.numel())

print(A.sum(axis=0))
print(A.sum(axis=0) / A.shape[0])
sum_A = A.sum(axis=1, keepdims=True)
print(sum_A)

print('*' * 20)
A = torch.arange(20).reshape((5, 4))
print(A)
print("A.cumsum(axis=0):")
print(A.cumsum(axis=0))

print('-' * 30)
print(A)
A = A.type(torch.FloatTensor)
X = torch.arange(4, dtype=torch.float32)
print(A.shape)
print(X.shape)
print(A, X)
print(torch.mv(A, X))

B = torch.ones(4, 3)
print(torch.mm(A, B))
#
u = torch.tensor([3., -4.])
print(torch.norm(u))
print(torch.abs(u).sum())
print('-'*30)
print(torch.norm(torch.ones((4, 9))))
print(torch.norm(u, p=1))
