# -*- coding: utf-8 -*-
# @Time : 2024/9/25 19:30
# @Author : nanji
# @Site : 
# @File : adaboost01.py
# @Software: PyCharm 
# @Comment : 
import cv2
import os

img = cv2.imread('data/1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
face_cascade.load('haarcascade_frontalface_alt2.xml')  # 一定要告诉编译器文件所在的
# 此文件是 opencv的haar人脸特征分类器
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
	img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)
	cv2.imshow('img', img)
	cv2.waitKey()
