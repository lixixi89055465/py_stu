'''

'''
import torch
t = torch.tensor([[[1, 2, 2, 1],
                   [3, 4, 4, 3],
                   [1, 2, 3, 4]],
                  [[5, 6, 6, 5],
                   [7, 8, 8, 7],
                   [5, 6, 7, 8]]])
print(t.shape)

x = torch.flatten(t, start_dim=1)
print(x.shape)

y = torch.flatten(t, start_dim=0, end_dim=1)
print(y.shape)