import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

print("Sklearn version is {}".format(sklearn.__version__))
import seaborn as sns

sns.set(style='darkgrid', font_scale=1.4)
from imblearn.over_sampling import SMOTE
import itertools
import warnings

warnings.filterwarnings('ignore')
import plotly.express as px
import time

# Sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, plot_confusion_matrix, plot_roc_curve, roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.utils import resample

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

print('Train set shape:', train.shape)
print('Test set shape:', test.shape)
print(train.head())
print('TRAIN SET MISSING VALUES:')
print(train.isna().sum())
print('')
print('TEST SET MISSING VALUES:')
print(test.isna().sum())
# print(f'Duplates in train set:{train.duplicated().sum()},({np.round(100 * train.duplicated().sum() / len(train), 1)}%)')

print(
    f'Duplicated in train set :{train.duplicated().sum()},({np.round(100 * train.duplicated().sum() / len(train), 1)}%)')
print(
    f'Duplicated in test set :{test.duplicated().sum()},({np.round(100 * train.duplicated().sum() / len(train), 1)}%)')
print(train.nunique())
print(train.dtypes)
# plt.figure(figsize=(6, 6,))
# train['Transported'].value_counts().plot.pie(explode=[0.1, 0.1], autopct='%1.1f%%', shadow=True,
#                                              textprops={'fontsize': 16}).set_title('Target distribution')

plt.figure(figsize=(10, 4))
# Histogram
# sns.histplot(data=train, x='Age', hue='Transported', binwidth=1, kde=True)
# plt.show()

# Expenditure features

# exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
# fig = plt.figure(figsize=(10, 20))
# for i, var_name in enumerate(exp_feats):
#     ax = fig.add_subplot(5, 2, 2 * i + 1)
#     sns.histplot(data=train, x=var_name, axes=ax, bins=30, kde=False, hue='Transported')
#     ax.set_title(var_name)
#     ax = fig.add_subplot(5, 2, 2 * i + 2)
#     sns.histplot(data=train, x=var_name, axes=ax, bins=30, kde=True, hue='Transported')
#     plt.ylim([0, 100])
#     ax.set_title(var_name)
#
# fig.tight_layout()
# plt.show()

# Categorical features
# cat_feats = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
# fig = plt.figure(figsize=(10, 16))
#
# for i, var_name in enumerate(cat_feats):
#     ax = fig.add_subplot(4, 1, i + 1)
#     sns.countplot(data=train, x=var_name, axes=ax, hue='Transported')
#     ax.set_title(var_name)
#
# fig.tight_layout()
# plt.show()

print('1' * 100)
qual_feats = ['PassengerId', 'Cabin', 'Name']
print(train[qual_feats].head())
# plt.show()
print('2' * 100)
train['Age_group'] = np.nan
train.loc[train['Age'] <= 12, 'Age_group'] = 'Age_0-12'
train.loc[(train['Age'] > 12) & (train['Age'] < 18), 'Age_group'] = 'Age_13-17'
train.loc[(train['Age'] >= 18) & (train['Age'] <= 25), 'Age_group'] = 'Age_18-25'
train.loc[(train['Age'] >= 26) & (train['Age'] <= 30), 'Age_group'] = 'Age_26-30'
train.loc[(train['Age'] > 30) & (train['Age'] <= 50), 'Age_group'] = 'Age_31-50'
train.loc[(train['Age'] > 50), 'Age_group'] = 'Age_51+'

test['Age_group'] = np.nan
test.loc[train['Age'] <= 12, 'Age_group'] = 'Age_0-12'
test.loc[(train['Age'] > 12) & (train['Age'] < 18), 'Age_group'] = 'Age_13-17'
test.loc[(train['Age'] >= 18) & (train['Age'] <= 25), 'Age_group'] = 'Age_18-25'
test.loc[(train['Age'] >= 26) & (train['Age'] <= 30), 'Age_group'] = 'Age_26-30'
test.loc[(train['Age'] > 30) & (train['Age'] <= 50), 'Age_group'] = 'Age_31-50'
test.loc[(train['Age'] > 50), 'Age_group'] = 'Age_51+'
# Plot distribution of new features

# plt.figure(figsize=(10, 4))
# g = sns.countplot(data=train, x='Age_group', hue='Transported',
#                   order=['Age_0-12', 'Age_13-17', 'Age_18-25', 'Age_26-30', 'Age_31-50', 'Age_51+'])
# plt.title('Age group distribution')
# plt.show()

# New features - training set
# Expenditure features
exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
train['Expenditure'] = train[exp_feats].sum(axis=1)
train['No_spending'] = (train['Expenditure'] == 0).astype(int)
# New features  - test set
test['Expenditure'] = test[exp_feats].sum(axis=1)
test['No_spending'] = (test['Expenditure']).astype(int)
# Plot distribution of new features

# fig = plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# sns.histplot(data=train, x='Expenditure', hue='Transported', bins=200)
#
# plt.title('Total expenditure (truncated) ')
# plt.ylim([0, 200])
# plt.xlim([0, 20000])
#
# plt.subplot(1, 2, 2)
# sns.countplot(data=train, x='No_spending', hue='Transported')
# plt.title('No spending indicator')
# fig.tight_layout()
# plt.show()
train['Group'] = train['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
test['Group'] = test['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)

# New feature - group size
train['Group_size'] = train['Group'].map(lambda x: pd.concat([train['Group'], test['Group']]).value_counts()[x])
test['Group_size'] = test['Group'].map(lambda x: pd.concat([train['Group'], test['Group']]).value_counts()[x])
a=pd.concat([train['Group'],test['Group']])
print('8'*100)
print(a.head())

#Plot distribution of new features
fig=plt.figure(figsize=(12,4))
plt.figure(figsize=(20,4))
plt.subplot(1,2,1)
sns.histplot(data=train,x='Group',hue='Transported',binwidth=1)
plt.title('Group')

plt.subplot(1,2,2)
sns.countplot(data=train, x='Group_size', hue='Transported')
plt.title('Group size')
fig.tight_layout()
plt.show()