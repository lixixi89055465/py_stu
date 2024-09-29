# -*- coding: utf-8 -*-
# @Time : 2024/9/28 11:15
# @Author : nanji
# @Site : 
# @File : facerec_from_video_file.py
# @Software: PyCharm 
# @Comment :
import face_recognition
import cv2

# Open the input movie file
# 读入影片并得到影片长度
input_movie = cv2.VideoCapture('1.mp4')
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
# Create an output movie file (make sure resolution / frame rate matches input video!
fourcc = cv2.VideoWriter.fourcc(*'XVID')
# 第一个参数是要保存的文件的路径
# fourcc 指定编码器
# fps 要保存的视频的帧率
# frameSize 要保存的文件的画面尺寸
# isColor 指示是黑白画面还是彩色的画面
# fourcc 本身是一个32位的无符号 数值，用4个字母表示采用的编码器。
# 常用的有"DIVX","MJPG","XVID","X264"。可用的列表在这里 .
# 推荐使用"XVID",但一般依据你的电脑环境安装了哪些编码器 .
output_movie = cv2.VideoWriter('output.avi', fourcc, 25, (640, 360))
# output_movie = cv2.VideoWriter('output.avi', fourcc, 29.97, (640, 360))
# load some sample pictures and learn how to recognize them.
lmm_image = face_recognition.load_image_file('Qidian.png')
lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]

al_image = face_recognition.load_image_file('Quxiaoxiao.png')
al_face_encoding = face_recognition.face_encodings(al_image)[0]

known_faces = [
	lmm_face_encoding,
	al_face_encoding
]
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
	ret, frame = input_movie.read()
	frame_number += 1
	if not ret:
		break
	rgb_frame = frame[:, :, ::-1]
	face_locations = face_recognition.face_locations(rgb_frame)
	face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
	face_names = []
	for face_encoding in face_encodings:
		match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.5)
		name = None
		if match[0]:
			name = 'Qidian'
		elif match[1]:
			name = 'Quxiaoxiao'
		face_names.append(name)
	for (top, right, bottom, left), name in zip(face_locations, face_names):
		if not name:
			continue
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
		cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
	print('writing frame {} / {} '.format(frame_number, length))
	output_movie.write(frame)
input_movie.release()
cv2.destroyAllWindows()
