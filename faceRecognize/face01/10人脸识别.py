# -*- coding: utf-8 -*-
# @Time : 2024/9/16 20:56
# @Author : nanji
# @Site : 
# @File : 10人脸识别.py
# @Software: PyCharm 
# @Comment : 
import warnings

warnings.filterwarnings('ignore')
import cv2
import os
import urllib
import urllib.request

# 加载训练数据集文件
recogizer = cv2.face.LBPHFaceRecognizer_create()

# 加载数据
recogizer.read('trainer/trainer.yml')
# 名称
names = []
# 警报全局 变量
warningtime = 8


# md5 加密
def md5(str):
	import hashlib
	m = hashlib.md5()
	m.update(str.encode('utf8'))
	return m.hexdigest()


# 短信反馈
statusStr = {
	'0': '0',
	'-1': '-1',
	'-2': '-2',
	'30': '30',
	'40': '40',
	'41': '41',
	'42': '42',
	'43': '43',
	'50': '50',
}


# 报警模块
def warning():
	smsapi = 'http://api.smsbao.com'
	# 短信平台账号
	user = '185***99@qq.com'
	# 短信平台密码
	password = md5("*****")
	# 要发送的短信内容
	content = '【报警】\n原因：xxx \n 地点：xxx \n 时间：xxx'
	# 要发送短信的手机密码
	phone = '1855*****'
	data = urllib.parse.urlencode({'u': user, 'p': password, 'm': phone, 'c': content})
	send_url = smsapi + 'sms?' + data
	response = urllib.request.urlopen(send_url)
	the_page = response.read().decode('utf-8')
	print(statusStr[the_page])


def face_detect_demo(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度
	face_detector = cv2.CascadeClassifier(
		'C:/anaconda3/envs/py39/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
	face = face_detector.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100), (300, 300))
	# face=face_detector.detectMultiScale(gray)
	for x, y, w, h in face:
		cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
		cv2.circle(img, center=(x + w // 2, y + h // 2), radius=w // 2, color=(0, 255, 0), thickness=1)
		# 人脸识别
		ids, confidence = recogizer.predict(gray[y:y + h, x:x + w])
		print('标签id:', ids, '置信评分:', confidence)
		if (confidence > 80):
			global warningtime
			warningtime += 1
			if warningtime > 100:
				# warning()
				print('id 结束了' )
				warningtime = 0
			cv2.putText(img, 'unknow', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
		else:
			cv2.putText(img, str(names[ids - 1]), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
	cv2.imshow('result', img)


def name():
	path = './data/jm/'
	imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
	for imagePath in imagePaths:
		name = str(os.path.split(imagePath)[1].split('.', 2)[1])
		names.append(name)


cap = cv2.VideoCapture('1.mp4')
name()
while True:
	flag, frame = cap.read()
	if not flag:
		break
	face_detect_demo(frame)
	if ord(' ') == cv2.waitKey(10):
		break
cv2.destroyAllWindows()
cap.release()
# print(names)
