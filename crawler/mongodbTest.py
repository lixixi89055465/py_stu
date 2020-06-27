# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/27 7:37
@file: mongodbTest.py
@desc: 
"""
import pymongo
client=pymongo.MongoClient('localhost')
db=client['newtestdb']
db['table'].insert({'name' : 'bob'})
print(db['table'].find_one({'name': 'bob'}))
