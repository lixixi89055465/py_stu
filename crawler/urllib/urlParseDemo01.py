# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 18:01
@file: urlParseDemo01.py
@desc: 
"""
from urllib.parse import urlparse

result = urlparse('http://www.baidu.com/index.html;user?id=5#comment')
print(type(result), result)
