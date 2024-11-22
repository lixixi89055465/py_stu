import cv2
import os
from PIL import Image
import numpy as np


def getImageAndLabels(path):
	# 存储人脸数据
	faceSamples = []
	# 存储姓名数据
	ids = []
	# 存储图片信息
	imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
	# 加载分类器
	face_detector = cv2.CascadeClassifier(
		'C:/anaconda3/envs/py39/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
	# 遍历列表中的图片
	for imagePath in imagePaths:
		# 打开图片
		PIL_img = Image.open(imagePath).convert('L')
		# 将图片转换为数组，已黑白深浅
		img_numpy = np.array(PIL_img, 'uint8')
		# 获取图片人脸特征
		faces = face_detector.detectMultiScale(img_numpy)
		# 获取每张图片的id和姓名
		id = int(os.path.split(imagePath)[1].split('.')[0])
		# 预防无面容照片
		for x, y, w, h in faces:
			ids.append(id)
			faceSamples.append(img_numpy[y:y + h, x:x + w])
	# 打印脸部特征和 id
	print('id:', id)
	print('fs:', faceSamples)
	return faceSamples, ids


if __name__ == '__main__':
	# 图片路径
	path = './data/jm/'
	# 获取图片数组和id标签数组和姓名
	faces, ids = getImageAndLabels(path)
	# 获取 识别器
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	# 训练
	recognizer.train(faces,np.array(ids))
	# 保存文件
	recognizer.write('trainer/trainer.yml')

