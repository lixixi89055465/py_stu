import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
import cv2
import timm

import albumentations
from albumentations import pytorch as AT

# below are all from https://github.com/seefun/TorchUtils, thanks seefun to provide such useful tools
# import torch_utils as tu

seed = 415
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


CLEAN_DATASET = 0
FOLD = 5
csv = pd.read_csv('clean_train_v4.csv') if CLEAN_DATASET else pd.read_csv(
	'train.csv')  # cleaned data has no obvious improvement
sfolder = StratifiedKFold(n_splits=FOLD, random_state=seed, shuffle=True)
train_folds = []
val_folds = []

for train_idx, val_idx in sfolder.split(csv['image'], csv['label']):
	train_folds.append(train_idx)
	val_folds.append(val_idx)
	print(len(train_idx), len(val_idx))
