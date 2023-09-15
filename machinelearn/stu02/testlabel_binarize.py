# -*- coding: utf-8 -*-
# @Time    : 2023/9/15 17:25
# @Author  : nanji
# @Site    : 
# @File    : testlabel_binarize.py
# @Software: PyCharm 
# @Comment : https://zhuanlan.zhihu.com/p/522750594
from sklearn.preprocessing import label_binarize

print('0' * 100)
print(label_binarize([1, 6], classes=[1, 2, 4, 6]))

print('1' * 100)
print(label_binarize(["yes", "no", "no", "yes"], classes=["no", "yes"]))
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
print('2' * 100)
y = ['paris', 'paris', 'tokyo', 'amsterdam']
y_le = le.fit_transform(y)
print(y_le)
print('3'*100)
array=label_binarize([1,6],classes=[1,2,4,6])
print(array)
print('4'*100)
array1=label_binarize([1,6],classes=[1,6,4,2])
print(array1)
from sklearn.preprocessing import LabelBinarizer
print('5'*100)
y_bina=LabelBinarizer.fit_transform(y)
print(y_bina)



