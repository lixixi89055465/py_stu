# -*- coding: utf-8 -*-
# @Time : 2024/1/11 21:40
# @Author : nanji
# @Site : https://www.bilibili.com/video/BV1Ca4y1t7DS?p=6&spm_id_from=pageDriver&vd_source=50305204d8a1be81f31d861b12d4d5cf
# @File : RandomForest.py
# @Software: PyCharm 
# @Comment :
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns

df = pd.read_csv('HR_comma_sep.csv', index_col=None)
print('0' * 100)
print(df.isnull().any())
print('1' * 100)
print(df.head())
df = df.rename(columns={'satisfaction_level': 'satisfaction',
						'last_evaluation': 'evaluation',
						'number_project': 'projectCount',
						'average_montly_hours': 'averageMonthlyHours',
						'time_spend_company': 'yearsAtCompany',
						'Work_accident': 'workAccident',
						'promotion_last_5years': 'promotion',
						'sales': 'department',
						'left': 'turnover'
						})
front = df['turnover']
df.drop(labels=['turnover'], axis=1, inplace=True)
df.insert(0, 'turnover', front)
print(df.head())
print('0' * 100)

print(df.shape)
print('1' * 100)
print(df.dtypes)

print('2' * 100)
print(df.turnover.value_counts())
tureover_rate = df.turnover.value_counts() / len(df)
print(tureover_rate)
print('3' * 100)
print(df.describe())

# 分组的平均数据统计
tureover_summary = df.groupby('turnover')
print('4' * 100)
print(tureover_summary.mean())
# 相关矩阵
corr = df.corr()
# sns.heatmap(
# 	corr,
# 	xticklabels=corr.columns.values,
# 	yticklabels=corr.columns.values,
# )
print('5' * 100)
# print(corr)

emp_population = df['satisfaction'][df['turnover'] == 0].mean()
emp_turnover_satisfaction = df[df['turnover'] == 1]['satisfaction'].mean()
print('未离职员工满意度:' + str(emp_population))
print('离职员工满意度:' + str(emp_turnover_satisfaction))

import scipy.stats as stats

print('6' * 100)
result = stats.ttest_1samp(a=df[df['turnover'] == 1]['satisfaction'], \
						   popmean=emp_population)
print(result)
print('7' * 100)

degree_freedom = len(df[df['turnover'] == 1])
# 临界值
LQ = stats.t.ppf(0.025, degree_freedom)  # 95%置信区间的左边界
RQ = stats.t.ppf(0.975, degree_freedom)  # 95%置信区间的右边界
print('The T-分布 左边界:' + str(LQ))
print('The T-分布 右边界:' + str(RQ))

import scipy.stats as stats

# 满意度的t-Test
result = stats.ttest_1samp(a=df[df['turnover'] == 1]['satisfaction'],  # 离职员工的满意度样本
						   popmean=emp_population)
print('8' * 100)
print(result)
fig = plt.figure(figsize=(15, 4))
ax = sns.kdeplot(df.loc[(df['turnover'] == 0), 'evaluation'], color='b', shade=True, label='no turnover')
ax = sns.kdeplot(df.loc[(df['turnover'] == 1), 'evaluation'], color='r', shade=True, label='turnover')
ax.set(xlabel='工作评价', ylabel='频率')
plt.title('工作评价的改了密度函数 -  离职 V.S。 未离职 ')

# 月平均工作时长改了密度估计
# fig = plt.figure(figsize=(15, 4))
# ax = sns.kdeplot(df.loc[(df['turnover'] == 0), 'averageMonthlyHours'], \
# 				 color='b', shade=True, \
# 				 label='no turnover')
# ax = sns.kdeplot(df.loc[(df['turnover'] == 1), 'averageMonthlyHours'], \
# 				 color='r', shade=True, \
# 				 label='turnover')
#
# ax.set(xlabel='月工作时长（时）', ylabel='频率')
# plt.title('月工作时长（时） - 离职 V.S. 未离职')
## 员工满意度概率

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, \
	precision_score, recall_score, confusion_matrix, precision_recall_curve

# 将 string类型转化为整形类型
# a = df['department'].astype('category').cat.codes
# print(a)

print('0' * 100)
df['department'] = df['department'].astype('category').cat.codes
df['salary'] = df['salary'].astype('category').cat.codes

target_names = 'turnover'
X = df.drop('turnover', axis=1)
y = df[target_names]
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.15, random_state=123, stratify=y
)
# print('1'*100)
# print(df.head())

from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO
from six import StringIO
from IPython.display import Image
import pydotplus
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
dtree = tree.DecisionTreeClassifier(
	criterion='entropy',
	min_weight_fraction_leaf=0.01
)
dtree = dtree.fit(X_train, y_train)
dt_roc_auc = roc_auc_score(y_test, dtree.predict(X_test))
print("决策树 AUC = %2.2f" % dt_roc_auc)
print(classification_report(y_test, dtree.predict(X_test)))

# 需安装GraphViz和pydotplus进行决策树的可视化
# 特征向量
feature_names = df.columns[1:]
# 文件缓存
dot_data = StringIO()
# 将决策树导入到dot中
export_graphviz(dtree, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_names,class_names=['0','1'])
# 将生成的dot文件生成graph
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# 将结果存入到png文件中
# graph.write_png('diabetes.png')
# 显示
# Image(graph.create_png())
# 获取特征重要性
importances = dtree.feature_importances_
# 获取特征名称
feat_names = df.drop(['turnover'],axis=1).columns
# 排序
indices = np.argsort(importances)[::-1]
# 绘图
plt.figure(figsize=(12,6))
plt.title("Feature importances by Decision Tree")
plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)
plt.xlim([-1, len(indices)])
plt.show()

