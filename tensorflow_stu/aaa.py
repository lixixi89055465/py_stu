#%%
import tensorflow as tf 
from tensorflow.keras import datasets,layers,optimizers 

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)

#%%
x=tf.random.normal([4,784]) 
x
#%%
net=layers.Dense(10) 
net.build((4,784)) 
net(x).shape
#%%
net.kernel.shape
#%%
