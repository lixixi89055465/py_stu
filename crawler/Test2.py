# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 11:34
@file: Test2.py
@desc: 
"""
import requests

response = requests.get('http://www.baidu.com')
print(response.text)
print(response.headers)
print(response.status_code)
headers = {
    'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Mobile Safari/537.36'
}
response = requests.get('http://www.baidu.com', headers=headers)
print(response.status_code)
print(response.content)
print('1'*100)
response=requests.get('https://dss0.bdstatic.com/5aV1bjqh_Q23odCf/static/superman/img/logo/bd_logo1-66368c33f8.png')
with open('./demo.jpg','wb') as f:
    f.write(response.content)


