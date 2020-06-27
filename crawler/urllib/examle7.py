# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 16:06
@file: examle7.py
@desc: 
"""
from urllib import request,parse
url='http://localhost/post'
headers={
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36',
    'Host':''
}
dict={
    'name':'Genery'
}
data=bytes(parse.urlencode(dict),encoding='utf-8')
req=request.Request(url=url,data=data,headers=headers,method="POST")
response=request.urlopen(req)
print(response.read().decode("utf-8"))
