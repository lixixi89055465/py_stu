import torch

x = torch.ones(1)
w = torch.full([1], 2.)
print(w)
mse = torch.nn.functional.mse_loss(torch.ones(1), x * w)
print(mse)
# a = torch.autograd.grad(mse, [w])
# print(a)

# w.requires_grad_()
# torch.autograd.grad(mse,[w])
mse = torch.nn.functional.mse_loss(torch.ones(1), x * w)
torch.autograd.grad(mse, [w])
print(mse)

# mse.backward()
mse = torch.nn.functional.mse_loss(torch.ones(1), x * w)
mse.backward()
print(w.grad)
