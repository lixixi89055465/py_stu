import torch

inputs=torch.rand(128,39)

target=torch.rand(128)
print(target.shape)

# print(target.unsqueeze(1).shape)
# target=target.unsqueeze(1)

print(target.repeat(1, 1).shape)