import torch
from torch import nn
loss_func_none = nn.CrossEntropyLoss(reduction="none")
loss_func_mean = nn.CrossEntropyLoss(reduction="mean")
loss_func_sum = nn.CrossEntropyLoss(reduction="sum")
pre = torch.tensor([[0.8, 0.5, 0.2, 0.5],
                    [0.2, 0.9, 0.3, 0.2],
                    [0.4, 0.3, 0.7, 0.1],
                    [0.1, 0.2, 0.4, 0.8]], dtype=torch.float)
tgt = torch.tensor([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=torch.float)
print(loss_func_none(pre, tgt))
print(loss_func_mean(pre, tgt))
print(loss_func_sum(pre, tgt))