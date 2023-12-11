import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2
import math

import torch
import torchvision
import timm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, cross_val_score
# Metric
from sklearn.metrics import f1_score, accuracy_score

# Augmentation
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

print(f'using device {device}')
