# -*- coding: utf-8 -*-
# @Time : 2024/9/25 22:03
# @Author : nanji
# @Site : 
# @File : find_facial_features_in_picture.py
# @Software: PyCharm 
# @Comment :
from PIL import Image, ImageDraw
import face_recognition

image = face_recognition.load_image_file("two_people.jpg")
face_landmarks_list = face_recognition.face_landmarks(image)
print(face_landmarks_list)

pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)

for face_landmarks in face_landmarks_list:
	for facial_feature in face_landmarks.keys():
		print('The {} in this face has the following points :{}'.format(facial_feature, face_landmarks[facial_feature]))
		d.line(face_landmarks[facial_feature], width=5)
		# for facial_feature in face_landmarks.keys():
		# 	d.line(face_landmarks[facial_feature], width=5)
pil_image.show()
