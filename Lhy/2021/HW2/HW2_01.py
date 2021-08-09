
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=":16:8"
import numpy as np
import torch
import torch.nn as nn
print('Loading images ...')


data_root='../images/timit_11/'
train = np.load(data_root + 'train_11.npy')
train_label = np.load(data_root + 'train_label_11.npy').astype(int)
test = np.load(data_root + 'test_11.npy')

print('Size of training images: {}'.format(train.shape))
print('Size of testing images: {}'.format(test.shape))