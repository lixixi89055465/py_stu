import torch

a = torch.rand(3)
print(a)
print(a.requires_grad_())

p = torch.nn.functional.softmax(a, dim=0)
print(p)

p.backward()
print(torch.autograd.grad(p[1], [a], retain_graph=True))

print(torch.autograd.grad(p[2], [a]))

