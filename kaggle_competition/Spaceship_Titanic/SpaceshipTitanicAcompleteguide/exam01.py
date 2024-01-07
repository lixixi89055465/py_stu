# -*- coding: utf-8 -*-
# @Time : 2024/1/3 22:02
# @Author : nanji
# @Site : https://www.kaggle.com/code/samuelcortinhas/spaceship-titanic-a-complete-guide
# @File : exam01.py
# @Software: PyCharm 
# @Comment : 
import os
import torch
from torch import nn
from d2l import torch as d2l

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

sns.set(style='darkgrid', font_scale=1.4)
from imblearn.over_sampling import SMOTE
import itertools
import warnings

warnings.filterwarnings('ignore')
import plotly.express as px
import time
# sklearn
from sklearn.model_selection import train_test_split, \
	GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, _plot, roc_curve
from sklearn.preprocessing import StandardScaler, \
	MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import eli5
from eli5.sklearn import PermutationImportance

# Model
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

