print("error!!!!!!")
import numpy as np
import pandas as pd
import copy
# visualize
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.templates.default = "plotly_dark"

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models, layers, Sequential

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

print(train_df.sample(3))
print(train_df.columns)
print(train_df.shape)
train_missing = pd.DataFrame(train_df.isna().sum()).sort_values(by=0, ascending=False)
test_missing = pd.DataFrame(test_df.isna().sum()).sort_values(by=0, ascending=False)
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# fig = make_subplots(1,2,column_titles=['train','test'],x_title='Missing Values')
#
# fig.add_trace(go.Bar(x=train_missing[0],y=train_missing.index,orientation="h",
#                      marker=dict(color=[n for n in range(12)])
#                     ),1,1
#              )
# fig.add_trace(go.Bar(x=test_missing[0],y=test_missing.index,orientation="h",
#                      marker=dict(color=[n for n in range(12)])
#                     ),1,2
#              )
# fig.update_layout(showlegend=False, title_text="Missing Values In Train&Test Set", title_x=0.5)
def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum() / data.isnull().count())
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    tt['Types'] = types
    return (tt)


missing_info = missing_data(train_df)
_template = dict(layout=go.Layout(font=dict(family='Frankling Gothic', size=12), width=1000))
# fig = px.box(train_df, y='RoomService', color='Transported', points='all', notched=True)
# fig.update_layout(template=_template, title='RoomService Distribution')
# fig.show()

num_impute_col = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
num_imputer = SimpleImputer(strategy='mean')
train_df[num_impute_col] = num_imputer.fit_transform(train_df[num_impute_col])
test_df[num_impute_col] = num_imputer.fit_transform(test_df[num_impute_col])

train_df.fillna(axis=0, method='ffill')
test_df.fillna(axis=0, method='ffill')


def calculate_total(df):
    total_cost = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
    df = df.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1, )
    df['Total_cost'] = total_cost
    return df


train_df = calculate_total(train_df)
test_df = calculate_total(test_df)
# fig=px.box(train_df,y='Total_cost',color='Transported',points='all',notched=True)
# fig.update_traces(quartilemethod='exclusive')
# fig.update_layout(template=_template,title='Total cost Distribution')
# fig.show()

label_series = train_df['Transported'].value_counts()
# print('*'*30)
# print(label_series.head())
# fig=go.Figure(go.Bar(x=label_series.index,y=label_series.values))
# fig.update_xaxes(title='Survive Or not')
# fig.update_yaxes(title='Count')

# fig=go.Figure()
# fig.add_trace(go.Histogram(x=train_df['Age'],name='train'))
# fig.add_trace(go.Histogram(x=test_df['Age'],name='test'))
# fig.update_layout(title_text='Age Distributon',xaxis_title_text='Age',yaxis_title_text='Count',barmode='stack')
# fig.show()

series1 = train_df.query('Transported==0')['Age']
series2 = train_df.query('Transported==1')['Age']

# fig=go.Figure()
# fig.add_trace(go.Histogram(x=series1,name='Transported:False'))
# fig.add_trace(go.Histogram(x=series2,name='Transported:True'))
#
# fig.update_layout(title_text='Transported Or Not Age Distributon',xaxis_title_text='Value',yaxis_title_text='Count',barmode='stack')
# fig.show()
label_cols = ["HomePlanet", "CryoSleep", "Cabin", "Destination", "VIP"]


def label_encoder(train_df, test_df, columns):
    for col in columns:
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)
        train_df[col] = LabelEncoder().fit_transform(train_df[col])
        test_df[col] = LabelEncoder().fit_transform(test_df[col])
    return train_df, test_df


train_df, test_df = label_encoder(train_df, test_df, label_cols)

train_df.drop(['Cabin', 'Name', 'PassengerId'], axis=1, inplace=True)
test_df.drop(['Cabin', 'Name', 'PassengerId'], axis=1, inplace=True)

X = train_df.drop('Transported', axis=1)
y = train_df['Transported']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

# model = models.Sequential([
#     layers.Dense(units=16, activation='relu', input_shape=[6]),
#     layers.Dense(units=32, activation='relu'),
#     layers.Dense(units=8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#     layers.Dense(units=1, activation='sigmoid')
# ])
model = models.Sequential([
    layers.Dense(units=16, activation='relu', input_shape=[6, ]),
    layers.Dense(units=32, activation='relu', ),
    layers.Dense(units=8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dense(units=1, activation='sigmoid')
])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# history = model.fit(x=X_train, y=y_train, batch_size=64, epochs=50, validation_data=(X_test, y_test))
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics='acc')
history=model.fit(x=X_train,y=y_train,batch_size=64,epochs=50,validation_data=(X_test,y_test))

his_df = pd.DataFrame(history.history)
hisfig = px.line(his_df, y=['acc', 'val_acc'], markers=True)
hisfig.show()

predictions = model.predict(test_df)
sub = pd.read_csv('../input/sample_submission.csv')
sub['Transported'] = predictions
sub['Transported'] = sub['Transported'].map(lambda x: True if x > 0.5 else False)
sub.to_csv('result.csv', index=True)
