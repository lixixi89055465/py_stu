# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 13:48
@file: example1.py
@desc: 
"""
import urllib.request
response=urllib.request.urlopen('http://www.baidu.com')
print(response.read().decode('utf-8'))