# -*- coding: utf-8 -*-
# @Time : 2024/1/6 22:20
# @Author : nanji
# @Site : https://blog.csdn.net/qq_41731517/article/details/102535860
# @File : testXGBClassifier.py
# @Software: PyCharm 
# @Comment :
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from xgboost import plot_importance

digits = datasets.load_iris()
## data analysis
print(digits.data.shape)
print(digits.target.shape)

x_train, x_test, y_train, y_test = train_test_split( \
	digits.data, digits.target, test_size=0.3, random_state=33)
model = XGBClassifier(
	learning_rate=0.1,
	n_estimators=100,
	max_depth=6,
	min_child_weight=1,
	gamma=0.,
	subsample=0.8,
	colsample_bytree=0.8,
	objective='multi:softmax',
	scale_pos_weight=1,
	random_state=27
)
model.fit(x_train, y_train, \
		  eval_set=[(x_test, y_test)], \
		  early_stopping_rounds=10, \
		  eval_metric='mlogloss', verbose=True)
fig, ax = plt.subplots(figsize=(15, 15))

# plot_importance(model, height=0.5, ax=ax, max_num_features=64)
# plt.show()
print('0' * 100)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('精确率:%.2f%%' % (accuracy * 100.0))
print('1' * 100)
import xgboost as xgb

xgb_param = model.get_xgb_params()
extra = {'num_class': 3}
xgb_param.update(extra)
xgtrain = xgb.DMatrix(x_train, label=y_train)
cvresult = xgb.cv(xgb_param, xgtrain,\

				  num_boost_round=5000,\
				  nfold=5, \
				  metrics=['mlogloss'], \
				  early_stopping_rounds=50,\
				  stratified=True, \
				  seed=1301)
# 交叉验证后最好的树
print('best number of trees = {}'.format(cvresult.shape[0]))
model.set_params(n_estimators=cvresult.shape[0])
# 把model的参数设置成最好的树对应的参数

# fig, ax = plt.subplots(figsize=(15, 15))
# plot_importance(model, height=0.5, ax=ax, max_num_features=64)
# plt.show()

### 预测
y_pred = model.predict(x_test)

### 模型正确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率: %.2f%%" % (accuracy * 100.0))
