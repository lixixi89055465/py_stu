import torch

cell1 = torch.nn.RNNCell(100, 20)
x = torch.randn(10, 3, 100)
h1 = torch.zeros(3, 20)
for xt in x:
    h1=cell1(xt,h1)
    print(h1.shape)
