# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 14:44
@file: example3.py
@desc: 
"""
import urllib.request
response=urllib.request.urlopen('https://www.python.org')
print(response)
print(response.read().decode("utf-8"))
