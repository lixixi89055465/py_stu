import numpy as np
import torch

data = torch.from_numpy(np.random.rand(3, 5))
print(str(data))

for i, data_i in enumerate(data.chunk(5, 1)):  # 沿1轴分为5块
    print(i,str(data_i))
