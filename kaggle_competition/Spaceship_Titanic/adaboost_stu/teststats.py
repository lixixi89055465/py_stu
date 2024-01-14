# -*- coding: utf-8 -*-
# @Time : 2024/1/13 14:05
# @Author : nanji
# @Site : https://www.cnblogs.com/HuZihu/p/12113253.html
# @File : teststats.py
# @Software: PyCharm 
# @Comment :
from scipy.stats import norm

critical1 = norm.ppf(0.95)  # 左尾或右尾
critical2 = norm.ppf(0.975)  # 双尾
print(critical1)
print(critical2)

print('0' * 100)
from scipy.stats import t

critical1 = t.ppf(0.95, 10)
critical2 = t.ppf(0.975, 10)
print(critical1)
print(critical2)
from scipy.stats import chi2

print('1' * 100)
critical1 = chi2.ppf(0.95, 10)  # 左尾或右尾
critical2 = chi2.ppf(0.975, 10)  # 双尾
print(critical1)
print(critical2)
print('2' * 100)
from scipy.stats import f

critical1 = f.ppf(0.95, 30, 28)  # 左尾或右尾
critical2 = f.ppf(0.975, 30, 28)  # 双尾
print(critical1)
print(critical2)
