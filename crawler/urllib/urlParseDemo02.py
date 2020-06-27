# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 18:03
@file: urlParseDemo02.py
@desc: 
"""
from urllib.parse import urlparse

result = urlparse('http://www.baidu.com/index.html;user?id=5#comment', scheme='https')
print(result)
