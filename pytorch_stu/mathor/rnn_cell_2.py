import torch

cell1 = torch.nn.RNNCell(100, 30)
cell2 = torch.nn.RNNCell(30, 20)
x = torch.randn(10, 3, 100)
h1 = torch.zeros(3, 30)
h2 = torch.zeros(3, 20)

for xt in x:
    h1 = cell1(xt, h1)
    h2 = cell2(h1, h2)
print(h2.shape)