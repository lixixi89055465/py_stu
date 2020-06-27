# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 18:11
@file: urlParseDemo03.py
@desc: 
"""
from urllib.parse import urlparse
result=urlparse('www.baidu.com/index.html;user?id=5#comment',scheme='https')
print(result)