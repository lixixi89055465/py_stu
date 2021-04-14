import torch

# uninitialized
print('-' * 30)
print(torch.empty([2, 3]))
print(torch.empty(1))
print(torch.IntTensor(2, 3))
print('-' * 20)
print(torch.FloatTensor(2, 3))
print(torch.tensor([1.2, 3]).type())
torch.set_default_tensor_type(torch.DoubleTensor)
print(torch.tensor([1.2, 3]).type())
