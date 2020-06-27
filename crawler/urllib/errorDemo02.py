# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 17:45
@file: errorDemo02.py
@desc: 
"""
from urllib import request, error

try:
    # response = request.urlopen('http://cuiqingcai.com/index.html')
    response = request.urlopen('http://www.cuiqingcai.com/index.html')
except error.HTTPError as e:
    print(e.reason, e.code, e.headers, sep='\n')
except error.URLError as e:
    print(e.reason)
else:
    print('Request Successfully')
