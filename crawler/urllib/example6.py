# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 15:43
@file: example6.py
@desc: 
"""
import urllib.request
response=urllib.request.urlopen('http://www.python.org')
print(response.status)
print(response.getheaders())
print(response.getheader('server'))