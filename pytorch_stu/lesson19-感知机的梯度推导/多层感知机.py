import torch

x = torch.randn(1, 10)
w = torch.randn(2, 10, requires_grad=True)

o = torch.sigmoid(x @ w.t())
print(o.shape)
loss = torch.nn.functional.mse_loss(torch.ones(1, 1).repeat(1,2), o)
print(loss)
loss.backward()

print(w.grad)