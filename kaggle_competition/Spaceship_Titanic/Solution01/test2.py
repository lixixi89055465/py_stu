import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_halving_search_cv
from sklearn.base import clone
from sklearn import datasets
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, classification_report, recall_score, \
    confusion_matrix, mean_squared_error, precision_score, recall_score, fbeta_score, f1_score, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, SGDRegressor, Lasso, Ridge, \
    ElasticNet
from sklearn.model_selection import train_test_split, KFold, cross_val_score, ShuffleSplit, cross_val_predict, \
    GridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV, HalvingGridSearchCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn import preprocessing
import re
import matplotlib as mpl
from pandas_profiling import ProfileReport

pd.set_option("display.precision", 6)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification, fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings

warnings.filterwarnings("ignore")
from sklearn.svm import SVC
from matplotlib import cm

## Pipline
# 1 - Data
# understanding < br >
# 2 - Data
# Exploration < br >
# 3 - Data
# cleaning < br >
# 4 - Modeling < br >
# 5 - Evaluation
# %%
import numpy as np

ser = pd.Series([390., 350., 30., 20.], index=['Falcon', 'Falcon', 'Parrot', 'Parrot'], name="Max Speed")
print(ser.groupby(["a", "b", "a", "b"]).mean())