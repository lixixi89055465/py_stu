# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 13:57
@file: example2.py
@desc: 
"""
import urllib.parse
import urllib.request
data=bytes(urllib.parse.urlencode({'world':'hello'}),encoding='utf-8')
response=urllib.request.urlopen('http://localhost/post',data=data)
print(response.read())