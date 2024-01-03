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
df = pd.read_csv('train.csv')
print(df.head())

df2 = df.drop(['PassengerId', 'Name', 'Cabin'], axis=1)
import numpy as np

df2['VIP'] = df2['VIP'].replace({'False': 0, 'True': 1}).astype(np.float16)
print(df2.groupby('HomePlanet')['Transported'].apply(lambda x: (x == 1).sum()).reset_index(name='Number Transported'))
count_transported = df2.groupby('HomePlanet')['Transported'].apply(lambda x: (x == 1).sum()).reset_index(
    name='Number Transported')
from matplotlib import cm


# cm.inferno_r(np.linspace(.4,.8,30))
# cm.inferno_r(np.linspace(.4, .5, 30))
# count_transported = count_transported.sort_values('Number Transported', ascending=False)
# count_transported.plot.bar(x='HomePlanet',y='Number Transported',figsize=(12,7))
# plt.xticks(rotation=0)
# plt.grid()
# plt.show()

# Data columns (total 11 columns):
#  #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   HomePlanet    8693 non-null   int64
#  1   CryoSleep     8476 non-null   float16
#  2   Destination   8693 non-null   int64
#  3   Age           8514 non-null   float64
#  4   VIP           8490 non-null   float16
#  5   RoomService   8512 non-null   float64
#  6   FoodCourt     8510 non-null   float64
#  7   ShoppingMall  8485 non-null   float64
#  8   Spa           8510 non-null   float64
#  9   VRDeck        8505 non-null   float64
#  10  Transported   8693 non-null   bool
# print('='*30)
# count_transported=df2.groupby("Destination")['Transported'].apply(lambda x:(x==1).sum()).reset_index(name='Number Transported')
# color=cm.inferno_r(np.linspace(0.4,0.8,30))
# count_transported=count_transported.sort_values('Number Transported',ascending=False)
# count_transported.plot.bar(x='Destination',y='Number Transported',color=color,figsize=(12,7))
# plt.xticks(rotation=0)
# plt.grid()
# plt.show()

def encode_categories(features):
    le = preprocessing.LabelEncoder()
    for i in range(len(features)):
        df2[features[i]] = le.fit_transform(df2[features[i]])


encode_categories(['HomePlanet', 'Destination'])
df2.drop(df[df['Age'] == 0].index, axis=0, inplace=True)
age_mean = df2['Age'].mean()
age_median = df2['Age'].median()
df2['Age'].fillna(age_mean, inplace=True)

df2.VIP.fillna(0, inplace=True)
df2['VIP'].fillna(0, inplace=True)
df2.fillna(0, inplace=True)
X_train = df2.drop('Transported', axis=1)
y_train = df2['Transported']
X_test = pd.read_csv('test.csv')
X_test['Cabin'].value_counts()
passenger = X_test['PassengerId']
X_test = X_test.drop(['PassengerId', 'Name', 'Cabin'], axis=1)

X_test['VIP'] = X_test['VIP'].replace({'False': 0, 'True': 1}).astype(np.float16)
X_test['CryoSleep'] = X_test['CryoSleep'].replace({'False': 0, 'True': 1}).astype(np.float16)

from sklearn import preprocessing


def encode_categories(features):
    le = preprocessing.LabelEncoder()
    for i in range(len(features)):
        X_test[features[i]] = le.fit_transform(X_test[features[i]])


encode_categories(['HomePlanet', 'Destination'])
age_mean = X_test['Age'].mean()
age_median = X_test['Age'].median()
X_test['Age'].fillna(age_mean, inplace=True)
X_test.fillna(0, inplace=True)
print(X_test.columns)

# 40
models = []
models.append(('LogisticRegression', LogisticRegression()))
models.append(('RandomForest', RandomForestClassifier()))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))

print('=' * 20)
for name, model in models:
    trained_model = model.fit(X_train, y_train)
    predictions = trained_model.predict(X_test)
    print(f"train score:{accuracy_score(y_train, trained_model.predict(X_train))}\n")

n_estimators = [5, 20, 50, 100]  # number of trees in the random forest
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 120, num=12)]
min_samples_split = [2, 6, 10]
min_samples_leaf = [1, 3, 4, 5, 6]
bootstrap = [True, False]

random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                               n_iter=100, cv=5, verbose=2, random_state=35, n_jobs=-1)

rf_random.fit(X_train, y_train)
# print(f"train score :{accuracy_score(y_train, rf_random.predict(X_train))}\n")
# print("Random grid:" ,random_grid,"\n")
print("Best parameters: ", rf_random.best_params_, '\n')
randmf = RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=6,
                                max_features='sqrt', max_depth=50, bootstrap=True)

result = randmf.fit(X_train, y_train)
print(f"train score:{accuracy_score(y_train, rf_random.predict(X_train))}\n")

y_prediction = randmf.predict(X_test)
data = {
    'PassengerId': passenger,
    'Transported': y_prediction
}
final = pd.DataFrame(data)
final.to_csv("test1.csv", index=False)
