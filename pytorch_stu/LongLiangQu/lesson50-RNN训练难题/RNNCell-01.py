import torch
import numpy as np

start = np.random.randint(3, size=1)[0]
time_steps=np.linspace(start,start+10,num_time_steps=10)
data=np.sin(time_steps)
