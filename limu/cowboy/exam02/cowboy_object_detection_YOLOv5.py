# Import lib
import os
import gc
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import yaml
from shutil import copyfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

json_file_path = './cowboyoutfits/train.json'

data = json.load(open(json_file_path, 'r'))
yolo_anno_path = './training/yolo_anno/'

if not os.path.exists(yolo_anno_path):
	os.makedirs(yolo_anno_path)
# 需要注意下因为我们的annotation lable是不连续的,会导致后面报错,所以这里生成一个map映射
cate_id_map = {}
num = 0
for cate in data['categories']:
	cate_id_map[cate['id']] = num
	num += 1

print(cate_id_map)

# 对比下
print(data['categories'])


# convert the bounding box from COCO to YOLO format.

def cc2yolo_bbox(img_width, img_height, bbox):
	dw = 1. / img_width
	dh = 1. / img_height
	x = bbox[0] + bbox[2] / 2.0
	y = bbox[1] + bbox[3] / 2.0
	w = bbox[2]
	h = bbox[3]

	x = x * dw
	w = w * dw
	y = y * dh
	h = h * dh
	return (x, y, w, h)


# transfer the annotation, and generated a train dataframe file
f = open('train.csv', 'w')
f.write('id,file_name\n')
for i in tqdm(range(len(data['images']))):
	filename = data['images'][i]['file_name']
	img_width = data['images'][i]['width']
	img_height = data['images'][i]['height']
	img_id = data['images'][i]['id']
	yolo_txt_name = filename.split('.')[0] + '.txt'  # remove .jpg

	f.write('{},{}\n'.format(img_id, filename))
	yolo_txt_file = open(os.path.join(yolo_anno_path, yolo_txt_name), 'w')

	for anno in data['annotations']:
		if anno['image_id'] == img_id:
			yolo_bbox = cc2yolo_bbox(img_width, img_height, anno['bbox'])  # "bbox": [x,y,width,height]
			yolo_txt_file.write(
				'{} {} {} {} {}\n'.format(cate_id_map[anno['category_id']], yolo_bbox[0], yolo_bbox[1], yolo_bbox[2],
										  yolo_bbox[3]))
	yolo_txt_file.close()
f.close()

# generate training dataframe
train = pd.read_csv('./train.csv')
print(train.head())

train_df, valid_df = train_test_split(train, test_size=0.1, random_state=233)
print(f'Size of total training images: {len(train)}, '
	  f'training images:{len(train_df)} . validation images: {len(valid_df)}')

# generate newtrain data frame with spliter mark
train_df.loc[:, 'split'] = 'train'
valid_df.loc[:, 'split'] = 'valid'
df = pd.concat([train_df, valid_df]).reset_index(drop=True)
print(df.sample(10))
# mdke directory for traning secti
# cd /kaggle/training/
# ls
os.makedirs('./training/cowboy/images/train', exist_ok=True)
os.makedirs('./training/cowboy/images/valid', exist_ok=True)

os.makedirs('./training/cowboy/labels/train', exist_ok=True)
os.makedirs('./training/cowboy/labels/valid', exist_ok=True)

# move the images and annotations to relevant splited folders

for i in tqdm(range(len(df))):
	row=df.loc[i]
	name=row.file_name.split('.')[0]
	if row.split=='train':
		copyfile(f'./cowboyoutfits/images/{name}.jpg',
				 f'./training/cowboy/images/train/{name}.jpg')
		copyfile(f'./training/yolo_anno/{name}.txt',
				 f'./training/cowboy/labels/train/{name}.txt')
	else:
		copyfile(f'./cowboyoutfits/images/{name}.jpg',
				 f'./training/cowboy/images/valid/{name}.jpg')
		copyfile(f'./training/yolo_anno/{name}.txt',
				 f'./training/cowboy/labels/valid/{name}.txt')

# Create yaml file
# cd ./training
data_yaml = dict(
	train='./training/cowboy/images/train/',
	val='./training/cowboy/images/valid/',
	nc=5,
	names=['belt', 'sunglasses', 'boot', 'cowboy_hat', 'jacket']
)
# we will make the file under the yolov5/data/ directory.
with open('./data/data.yaml', 'w') as outfile:
	yaml.dump(data_yaml, outfile, default_flow_style=True)

lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.2  # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937  # SGD momentum/Adam beta1
weight_decay: 0.0005  # optimizer weight decay 5e-4
warmup_epochs: 3.0  # warmup epochs (fractions ok)
warmup_momentum: 0.8  # warmup initial momentum
warmup_bias_lr: 0.1  # warmup initial bias lr
box: 0.05  # box loss gain
cls: 0.5  # cls loss gain
cls_pw: 1.0  # cls BCELoss positive_weight
obj: 1.0  # obj loss gain (scale with pixels)
obj_pw: 1.0  # obj BCELoss positive_weight
iou_t: 0.20  # IoU training threshold
anchor_t: 4.0  # anchor-multiple threshold
# anchors: 3  # anchors per output layer (0 to ignore)
fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4  # image HSV-Value augmentation (fraction)
degrees: 0.0  # image rotation (+/- deg)
translate: 0.1  # image translation (+/- fraction)
scale: 0.5  # image scale (+/- gain)
shear: 0.0  # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
flipud: 0.0  # image flip up-down (probability)
fliplr: 0.5  # image flip left-right (probability)
mosaic: 1.0  # image mosaic (probability)
mixup: 0.0  # image mixup (probability)
copy_paste: 0.0  # segment copy-paste (probability)

# IMG_SIZE = 640  # the default image size in yolo is 640, it will automated resize our image during training and valudation.
BATCH_SIZE = 32  # wisely choose, use the largest size that can feed up all your gpu ram
EPOCHS = 5
MODEL = 'yolov5m.pt'  # 5s, 5m 5l
name = f'{MODEL}_BS_{BATCH_SIZE}_EP_{EPOCHS}'

'''
python train.py --batch 32 \
                 --epochs 5 \
                 --data data.yaml \
                 --weights  yolov5m.pt \
                 --save_period 1 \
                 --project ./working/kaggle-cwoboy \
                 --name yolov5m.pt_BS_32_EP_5 \
                 --cache-images
                 
python ./train.py --batch 16 \
                 --epochs 5 \
                 --data ./data/data.yaml \
                 --weights  yolov5m.pt \
                 --project ./working/kaggle-cwoboy \
                 --name yolov5m.pt_BS_32_EP_5 
                 
                 
/home/sdb2/aidata/workspace/py_stu/limu/cowboy/exam02/training/cowboy/labels/train.cache               
/home/sdb2/aidata/workspace/py_stu/limu/cowboy/exam02/training/cowboy/labels/train.cache

/home/sdb2/aidata/workspace/py_stu/limu/data/cowboyoutfits/images/valid
 /home/sdb2/aidata/workspace/py_stu/limu/data/cowboyoutfits
 /home/sdb2/aidata/workspace/py_stu/limu/cowboy/exam02/yolov5/working/kaggle-cwoboy/yolov5m.pt_BS_32_EP_524/weights/best.pt
                 '''

valid_df = pd.read_csv('./cowboyoutfits/valid.csv')
test_df = pd.read_csv('./cowboyoutfits/test.csv')
print(valid_df.head())
print(valid_df.shape)
os.makedirs('./inference/valid', exist_ok=True)
os.makedirs('./inference/test', exist_ok=True)
# copy the validation image to inference folder for detection process
for i in tqdm(range(len(valid_df))):
    row = valid_df.loc[i]
    name = row.file_name.split('.')[0]
    copyfile(f'./cowboyoutfits/images/{name}.jpg', f'./inference/valid/{name}.jpg')

VALID_PATH = './inference/valid/'
MODEL_PATH = './cowboy-object-detection-models/v0_ep20_best.pt'
IMAGE_PATH = './cowboyoutfits/images/'

'''
python detect.py --weights ./working/kaggle-cwoboy/yolov5m.pt_BS_32_EP_524/weights/best.pt \
                  --source ./inference/valid/ \
                  --conf 0.546 \
                  --iou-thres 0.5 \
                  --save-txt \
                  --save-conf \
                  --augment
                  '''