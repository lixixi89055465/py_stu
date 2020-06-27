# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 18:18
@file: urlParseDemo05.py
@desc: 
"""
from urllib.parse import urlparse
result=urlparse('http://www.baidu.com/index.html#comment',allow_fragments=False)
print(result)

