# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 15:33
@file: example4.py
@desc: 
"""
import urllib.request
response=urllib.request.urlopen("http://localhost/get",timeout=1)
print(response.read())

