# -*- coding: utf-8 -*-
# @Time    : 2024/11/17 下午1:47
# @Author  : nanji
# @Site    : 
# @File    : test04.py
# @Software: PyCharm 
# @Comment :https://blog.csdn.net/qq_37281522/article/details/85032470

import random

list = [1, 2, 3]
print(random.sample(list, 2))

list = ["china", "python", "sky"]
print(random.sample(list, 2))

list = range(1, 10000)
print(random.sample(list, 5))
