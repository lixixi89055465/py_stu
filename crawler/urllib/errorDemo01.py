# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 17:41
@file: errorDemo01.py
@desc: 
"""
from urllib import request, error

try:
    response = request.urlopen('http://cuiqingcai.com/index.html')
except error.URLError as e:
    print(e.reason)