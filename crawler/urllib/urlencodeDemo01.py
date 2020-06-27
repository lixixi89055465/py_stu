# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 18:48
@file: urlencodeDemo01.py
@desc: 
"""
from urllib.parse import urlencode
params={
    'name':'germey',
    'age':22
}
base_url='http://www.baidu.com?'
url=base_url+urlencode(params)
