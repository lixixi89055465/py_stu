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
sns.heatmap(
	corr,
	xticklabels=corr.columns.values,
	yticklabels=corr.columns.values,
)
print('5' * 100)
print(corr)

emp_population = df['satisfaction'][df['turnover'] == 0].mean()
emp_population_satisfaction = df[df['turnover'] == 1]['satisfaction'].mean()
print('未离职员工满意度:' + str(emp_population))
print('离职员工满意度:' + str(emp_population_satisfaction))
import scipy.stats as stats

stats.ttest_1samp(a=df[df['turnover'] == 1]['satisfaction'], \
				  popmean=emp_population)

