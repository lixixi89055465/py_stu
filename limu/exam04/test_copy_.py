import torch

a = torch.tensor([1.,2.,3.],requires_grad=True)
b = a.clone()
print(a.data_ptr())
print(b.data_ptr())