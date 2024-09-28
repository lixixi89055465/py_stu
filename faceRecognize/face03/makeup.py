# -*- coding: utf-8 -*-
# @Time : 2024/9/28 9:56
# @Author : nanji
# @Site : 
# @File : makeup.py
# @Software: PyCharm 
# @Comment : 自动上妆
from PIL import Image, ImageDraw
import face_recognition

image = face_recognition.load_image_file('two_people.jpg')
face_landmarks_list = face_recognition.face_landmarks(image)
pil_image = Image.fromarray(image)
for face_landmarks in face_landmarks_list:
	d = ImageDraw.Draw(pil_image, 'RGBA')
	d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
	d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
	d.line(face_landmarks['left_eyebrow'], fill=(1, 1, 1, 1), width=15)
	d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)
	# eyebrow
	d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
	d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
	d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 128), width=8)
	d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128), width=1)
	# eye
	d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
	d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))
pil_image.show()

