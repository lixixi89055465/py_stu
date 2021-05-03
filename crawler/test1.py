# encoding: utf-8
"""
@author: nanjixiong
@time: 2020/6/26 21:49
@file: TextCNN.py
@desc: 
"""
from selenium import webdriver
import lxml
from bs4 import BeautifulSoup
import pyquery
import pymysql
import pymongo
driver=webdriver.PhantomJS()
driver.get('http://www.baidu.com')
soup=BeautifulSoup('<html></html>','lxml')
