nums = [1, 2, 3, 4]
targets = [False, True, False, True]

print([i for i in range(len(targets)) if targets[i]])
import numpy as np

import torch

target = torch.rand(2, 3)
arr = torch.rand(2, 3)

# print(targets.shape)
# print(arr.shape)

# print(torch.cat(targets, arr,dim=1))
print(torch.__version__)
print(type(nums))
