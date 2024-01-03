from IPython.display import clear_output

clear_output()
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split

from lightgbm import LGBMClassifier
import lazypredict
from lazypredict.Supervised import LazyClassifier
import time
import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')

RANDOM_STATE = 12
FOLDS = 5
STRATEGY = 'median'
'''
- `PassengerId` - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group 
    the passenger is travelling with and pp is their number within the group. People in a group are often 
    family members, but not always.
- `HomePlanet` - The planet the passenger departed from, typically their planet of permanent residence.
- `CryoSleep` - Indicates whether the passenger elected to be put into suspended animation for the duration 
    of the voyage. Passengers in cryosleep are confined to their cabins.
- `Cabin` - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be 
    either P for Port or S for Starboard.
- `Destination` - The planet the passenger will be debarking to.
- `Age` - The age of the passenger.
- `VIP` - Whether the passenger has paid for special VIP service during the voyage.
- `RoomService`, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the 
    Spaceship Titanic's many luxury amenities.
- `Name` - The first and last names of the passenger.
- `Transported` - Whether the passenger was transported to another dimension. This is the target, the column 
    you are trying to predict.

'''
# print(train.head())
# print(f'\033[94mNumber of rows in train input:{train.shape[0]}')
# print(f'\033[94mNumber of columns in train input:{train.shape[1]}')
# print(f'\033[94mnumber of values in train input:{train.count().sum()}')
# print(f'\033[94mnumber of missing values in train input:{sum(train.isna().sum())}')
# print(train.isna().sum().sort_values(ascending=False))
# print(train.describe())
#
# print(test.head())
# print(f'\033[94mNumber of rows in train input:{test.shape[0]}')
# print(f'\033[94mNumber of columns in train input:{test.shape[1]}')
# print(f'\033[94mnumber of values in train input:{test.count().sum()}')
# print(f'\033[94mnumber of missing values in train input:{sum(test.isna().sum())}')
#
# print(test.isna().sum().sort_values(ascending=False))
#
# print(test.describe())
# print(submission.head())

train.drop(['PassengerId'], axis=1, inplace=True)
test.drop(['PassengerId'], axis=1, inplace=True)
TARGET = 'PassengerId'
FEATURES = [col for col in train.columns if col != TARGET]
RANGE_STATE = 12

# train.iloc[:,:-1].describe().T.sort_values(by='std',ascending=False)\
#     .style.background_gradient(cmap='GnBu')\
#     .bar(subset=['max'],color='#BB0000')\
#     .bar(subset=['mean'],color='green')

test_null = pd.DataFrame(test.isna().sum())
test_null = test_null.sort_values(by=0, ascending=False)
train_null = pd.DataFrame(train.isna().sum())
train_null = train_null.sort_values(by=0, ascending=False)
# fig = make_subplots(rows=1,
#                     cols=2,
#                     column_titles=["Train Data", "Test Data"],
#                     x_title='Missing Values')
#
# fig.add_trace(go.Bar(
#     x=train_null[0],
#     y=train_null.index,
#     orientation='h',
#     marker=dict(
#         color=[n for n in range(12)],
#         line_color='rgb(0,0,0)',
#         line_width=2,
#         coloraxis='coloraxis')),
#     1, 1)
#
# fig.add_trace(go.Bar(x=test_null[0],
#                      y=test_null.index,
#                      orientation='h',
#                      marker=dict(
#                          color=[n for n in range(12)],
#                          line_color='rgb(0,0,0)',
#                          line_width=2,
#                          coloraxis='coloraxis')), 1, 2)
# fig.update_layout(showlegend=False, title_text="Column wise null value distribution", title_x=0.5)
# print('1'*40)
# fig.show()

missing_train_row = train.isna().sum(axis=1)
missing_train_row = pd.DataFrame(missing_train_row.value_counts() / train.shape[0]).reset_index()
missing_test_row = test.isna().sum(axis=1)
missing_test_row = pd.DataFrame(missing_test_row.value_counts() / test.shape[0]).reset_index()
missing_train_row.columns=['no','count']
missing_test_row.columns=['no','count']
missing_train_row['count']=missing_train_row['count']*100
missing_test_row['count']=missing_test_row['count']*100

fig = make_subplots(rows=1,
                    cols=2,
                    column_titles = ["Train Data", "Test Data"] ,
                    x_title="Missing Values",)

fig.add_trace(go.Bar(x=missing_train_row["no"],
                     y=missing_train_row["count"]  ,
                    marker=dict(color=[n for n in range(4)],
                                line_color='rgb(0,0,0)' ,
                                line_width = 3
                                ,coloraxis="coloraxis")),
              1, 1)
fig.add_trace(go.Bar(x= missing_test_row["no"],
                     y=missing_test_row["count"],
                    marker=dict(color=[n for n in range(4)],
                                line_color='rgb(0,0,0)',
                                line_width = 3 ,
                                coloraxis="coloraxis")),
              1, 2)
fig.update_layout(showlegend=False, title_text="Row wise Null Value Distribution", title_x=0.5)