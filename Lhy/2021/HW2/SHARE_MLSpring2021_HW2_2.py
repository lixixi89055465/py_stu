import numpy as np
from math import pi
from collections import defaultdict

from autograd_lib import autograd_lib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import warnings

warnings.filterwarnings('ignore')

student_id = 'nanji'  # fill with your student ID


class MathRegressor(nn.Module):
    def __init__(self, num_hidden=128):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(1, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1)
        )

    def forward(self, x):
        x = self.regressor(x)
        return x


import re

key = student_id[-1]
if re.match('[0-9]', key) is not None:
    key = int(key)
else:
    key = ord(key) % 10
# load checkpoint and images corresponding to the key

model = MathRegressor()
autograd_lib.register(model)
data = torch.load('../images/images.pth')[key]

model.load_state_dict(data['model'])
train, target = data['images']


# function to compute gradient norm

def compute_gradient_norm(model, criterion, train, target):
    model.train(0)
    model.zero_grad()
    output = model(train)
    loss = criterion(output, target)
    loss.backward()
    grads = []
    for p in model.regressor.children():
        if isinstance(p, nn.Linear):
            param_norm = p.weight.grad.norm(2).item()
            grads.append(param_norm)
    grad_mean = np.mean(grads)
    return grad_mean
