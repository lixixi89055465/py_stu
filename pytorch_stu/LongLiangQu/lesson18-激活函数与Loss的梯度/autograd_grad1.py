import torch

x = torch.ones(1)
w = torch.full([1], 2.)
w.requires_grad_()
mse = torch.nn.functional.mse_loss(torch.ones(1), x * w)
mse.backward()
print(w.grad)
print(x.shape)
print(w.shape)