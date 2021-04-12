import torch

import numpy  as np

a = np.array([2, 3.3])
print(a)
print(torch.from_numpy(a))
print(np.ones([2, 3]))
a=np.ones([2,3])
print(torch.from_numpy(a))
print(a.dtype)
# import form list

print(torch.tensor([2., 3.2]))
print('-'*30)
print(torch.FloatTensor([2., 3.2]))
print(torch.tensor([[2., 3.2], [1., 22.3]]))
print(torch.Tensor([2, 3]))
print(torch.FloatTensor([2, 3, 4, 54]))
print(torch.FloatTensor(2, 3))
print(torch.Tensor(4, 5))
# uninitialized
print('-'*30)
print(torch.empty([2, 3]))
