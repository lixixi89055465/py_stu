# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 18:44
@file: urlunparseDemo01.py
@desc: 
"""
from urllib.parse import urlunparse

data = ['http', 'www.baidu.com', 'index.html', 'uesr', 'a=6', 'comment']
print(urlunparse(data))
