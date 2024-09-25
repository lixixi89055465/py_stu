# -*- coding: utf-8 -*-
# @Time : 2024/9/16 22:36
# @Author : nanji
# @Site : 
# @File : 11网页视频.py
# @Software: PyCharm 
# @Comment : 
import warnings

warnings.filterwarnings('ignore')
import cv2


class CaptureVideo(object):
	def net_video(self):
		# 获取网络视频流
		# cam = cv2.VideoCapture("rtmp://58.200.131.2:1935/livetv/cctv5")
		# 伊朗
		cam = cv2.VideoCapture("rtmp://ns8.indexforce.com/home/mystream")
		# cam = cv2.VideoCapture("rtmp://mobliestream.c3tv.com:554/live/goodtv.sdp")
		while cam.isOpened():
			sucess, frame = cam.read()
			cv2.imshow('NetWork', frame)
			cv2.waitKey(1)


if __name__ == '__main__':
	capture_video = CaptureVideo()
	capture_video.net_video()
