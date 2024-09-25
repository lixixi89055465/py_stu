# -*- coding: utf-8 -*-
# @Time : 2024/9/25 20:14
# @Author : nanji
# @Site : 
# @File : adaboost_loop.py
# @Software: PyCharm 
# @Comment :
import cv2
import os

datapath = 'data/'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
face_cascade.load('haarcascade_frontalface_alt2.xml')

for img in os.listdir(datapath):
	print('1')
	frame = cv2.imread(datapath+img)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x, y, w, h) in faces:
		frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
		# cv2.imshow('img', frame)
		# cv2.waitKey()
	cv2.imwrite('result/' + img, frame)
