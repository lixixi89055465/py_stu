import torch
m = torch.nn.Identity(5, unused_argument1=0.1, unused_argument2=False)
input = torch.randn(128, 20)
output = m(input)
print(output.size())
print(m)