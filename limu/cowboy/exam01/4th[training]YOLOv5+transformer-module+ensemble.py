import os
import shutil
import json
import yaml
import random
import pandas as pd

# !pip install -q --upgrade wandb
# key=YOUR_WANDB_TOKEN
# import wandb
# wandb.login(key=key)
# !wandb off

data = json.load(open('../../data/cowboyoutfits/train.json', 'r'))
ann = data['annotations']
random.seed(34)
random.shuffle(ann)

ci = [87, 1034, 131, 318, 588]  # category_id, 分别对应belt,sunglasses,boot,cowboy_hat,jacket

print('total:')
for i in ci:
	count = 0
	for j in ann:
		if j['category_id'] == i:
			count += 1
	print(f'id: {i} counts: {count}')

total_id = set(each['image_id'] for each in ann)
val_id = set()
a, b, c, d, e = 0, 0, 0, 0, 0  # 用于每类的计数
for each in ann:
	if (each['category_id'] == ci[0]) and (a < 2):
		val_id.add(each['image_id'])
		a += 1
	elif (each['category_id'] == ci[1]) and (b < 20):
		val_id.add(each['image_id'])
		b += 1
	elif (each['category_id'] == ci[2]) and (c < 4):
		val_id.add(each['image_id'])
		c += 1
	elif (each['category_id'] == ci[3]) and (d < 7):
		val_id.add(each['image_id'])
		d += 1
	elif (each['category_id'] == ci[4]) and (e < 17):
		val_id.add(each['image_id'])
		e += 1

val_ann = []
for imid in val_id:
	for each_ann in ann:
		if each_ann['image_id'] == imid:
			val_ann.append(each_ann)

print(len(val_id), len(val_ann))

print('val set:')
for kind in ci:
	num = 0
	for i in val_ann:
		if i['category_id'] == kind:
			num += 1
	print(f'id: {kind} counts: {num}')

# The rest images are for training
train_id = total_id - val_id
train_ann = []
for each_ann in ann:
	for tid in train_id:
		if each_ann['image_id'] == tid:
			train_ann.append(each_ann)
			break
len(train_id), len(train_ann)

os.makedirs('../../data/cowboyoutfits/images/train', exist_ok=True)
os.makedirs('../../data/cowboyoutfits/images/val', exist_ok=True)

train_img = []
# Move train images
for j in data['images']:
	for i in train_id:
		if j['id'] == i:
			shutil.copy('../../data/cowboyoutfits/images/' + j['file_name'], '../../data/cowboyoutfits/train')
			train_img.append(j)

val_img = []
# Move val images
for j in data['images']:
	for i in val_id:
		if j['id'] == i:
			shutil.copy('../../data/cowboyoutfits/images/' + j['file_name'], '../../data/cowboyoutfits/val')
			val_img.append(j)

print('0' * 100)
print(len(val_img), len(train_img))

os.makedirs('../../data/cowboyoutfits/labels/train', exist_ok=True)
os.makedirs('../../data/cowboyoutfits/labels/val', exist_ok=True)

train_info = [(each['id'], each['file_name'].split('.')[0], each['width'], each['height']) for each in train_img]
val_info = [(each['id'], each['file_name'].split('.')[0], each['width'], each['height']) for each in val_img]

trans = {f'{each}': f'{idx}' for (idx, each) in enumerate(ci)}  # Mapping the category_ids

# Create *.txt files for training
for (imid, fn, w, h) in train_info:
	with open('../../data/cowboyoutfits/labels/train/' + fn + '.txt', 'w') as t_f:
		for t_ann in train_ann:
			if t_ann['image_id'] == imid:
				# convert X_min,Y_min,w,h to X_center/width,Y_center/height,w/width,h/height
				bbox = [str((t_ann['bbox'][0] + (t_ann['bbox'][2] / 2) - 1) / float(w)) + ' ',
						str((t_ann['bbox'][1] + (t_ann['bbox'][3] / 2) - 1) / float(h)) + ' ',
						str(t_ann['bbox'][2] / float(w)) + ' ',
						str(t_ann['bbox'][3] / float(h))]
				t_f.write(trans[str(t_ann['category_id'])] + ' ' + str(bbox[0] + bbox[1] + bbox[2] + bbox[3]))
				t_f.write('\n')

# Create *.txt files for evaluating
for (imid, fn, w, h) in val_info:
	with open('../../data/cowboyoutfits/labels/val/' + fn + '.txt', 'w') as v_f:
		for v_ann in val_ann:
			if v_ann['image_id'] == imid:
				# convert X_min,Y_min,w,h to X_center/width,Ycenter/height,w/width,h/height
				bbox = [str((v_ann['bbox'][0] + (v_ann['bbox'][2] / 2) - 1) / float(w)) + ' ',
						str((v_ann['bbox'][1] + (v_ann['bbox'][3] / 2) - 1) / float(h)) + ' ',
						str(v_ann['bbox'][2] / float(w)) + ' ',
						str(v_ann['bbox'][3] / float(h))]
				v_f.write(trans[str(v_ann['category_id'])] + ' ' + str(bbox[0] + bbox[1] + bbox[2] + bbox[3]))
				v_f.write('\n')

data_yaml = dict(
	path='../../data/cowboyoutfits',  # dataset root dir
	train='train',  # train images (relative to 'path')
	val='val',  # val images (relative to 'path')
	test='test',  # test images (relative to 'path') 推理的时候才用得到
	nc=5,
	names=['belt', 'sunglasses', 'boot', 'cowboy_hat', 'jacket'],
	download='None'
)
with open('./my_data_config.yaml', 'w') as f:
	yaml.dump(data_yaml, f, default_flow_style=False)

hyp_yaml = dict(
	lr0=0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
	lrf=0.16,  # final OneCycleLR learning rate (lr0 * lrf)
	momentum=0.937,  # SGD momentum/Adam beta1
	weight_decay=0.0005,  # optimizer weight decay 5e-4
	warmup_epochs=5.0,  # warmup epochs (fractions ok)
	warmup_momentum=0.8,  # warmup initial momentum
	warmup_bias_lr=0.1,  # warmup initial bias lr
	box=0.05,  # box loss gain
	cls=0.3,  # cls loss gain
	cls_pw=1.0,  # cls BCELoss positive_weight
	obj=0.7,  # obj loss gain (scale with pixels)
	obj_pw=1.0,  # obj BCELoss positive_weight
	iou_t=0.20,  # IoU training threshold
	anchor_t=4.0,  # anchor-multiple threshold
	fl_gamma=0.0,  # focal loss gamma (efficientDet default gamma=1.5)
	hsv_h=0.015,  # image HSV-Hue augmentation (fraction)
	hsv_s=0.7,  # image HSV-Saturation augmentation (fraction)
	hsv_v=0.4,  # image HSV-Value augmentation (fraction)
	degrees=0.0,  # image rotation (+/- deg)
	translate=0.1,  # image translation (+/- fraction)
	scale=0.25,  # image scale (+/- gain)
	shear=0.0,  # image shear (+/- deg)
	perspective=0.0,  # image perspective (+/- fraction), range 0-0.001
	flipud=0.0,  # image flip up-down (probability)
	fliplr=0.5,  # image flip left-right (probability)
	mosaic=1.0,  # image mosaic (probability)
	mixup=0.0,  # image mixup (probability)
	copy_paste=0.0  # segment copy-paste (probability)
)
with open('./my_hyp_config.yaml', 'w') as f:
	yaml.dump(hyp_yaml, f, default_flow_style=False)

model1_yaml = dict(
	nc=5,  # numberof classes
	depth_multiple=1.33,  # model depth multiple
	width_multiple=1.25,  # layer channel multiple
	anchors=3,  # 这里把默认的anchors配置改成了3以启用autoanchor, 获取针对自己训练时的img_size的更优质的anchor size
	# YOLOv5 backbone
	backbone=
	# [from, number, module, args]
	[[-1, 1, 'Focus', [64, 3]],  # 0-P1/2
	 [-1, 1, 'Conv', [128, 3, 2]],  # 1-P2/4
	 [-1, 3, 'C3', [128]],
	 [-1, 1, 'Conv', [256, 3, 2]],  # 3-P3/8
	 [-1, 9, 'C3', [256]],
	 [-1, 1, 'Conv', [512, 3, 2]],  # 5-P4/16
	 [-1, 9, 'C3', [512]],
	 [-1, 1, 'Conv', [768, 3, 2]],  # 7-P5/32
	 [-1, 3, 'C3', [768]],
	 [-1, 1, 'Conv', [1024, 3, 2]],  # 9-P6/64
	 [-1, 1, 'SPP', [1024, [3, 5, 7]]],
	 [-1, 3, 'C3', [1024, 'False']],  # 11
	 ],

	# YOLOv5 head
	head=
	[[-1, 1, 'Conv', [768, 1, 1]],
	 [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
	 [[-1, 8], 1, 'Concat', [1]],  # cat backbone P5
	 [-1, 3, 'C3', [768, 'False']],  # 15

	 [-1, 1, 'Conv', [512, 1, 1]],
	 [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
	 [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
	 [-1, 3, 'C3', [512, 'False']],  # 19

	 [-1, 1, 'Conv', [256, 1, 1]],
	 [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
	 [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
	 [-1, 3, 'C3', [256, 'False']],  # 23 (P3/8-small)

	 [-1, 1, 'Conv', [256, 3, 2]],
	 [[-1, 20], 1, 'Concat', [1]],  # cat head P4
	 [-1, 3, 'C3', [512, 'False']],  # 26 (P4/16-medium)

	 [-1, 1, 'Conv', [512, 3, 2]],
	 [[-1, 16], 1, 'Concat', [1]],  # cat head P5
	 [-1, 3, 'C3', [768, 'False']],  # 29 (P5/32-large)

	 [-1, 1, 'Conv', [768, 3, 2]],
	 [[-1, 12], 1, 'Concat', [1]],  # cat head P6
	 [-1, 3, 'C3', [1024, 'False']],  # 32 (P6/64-xlarge)

	 [[23, 26, 29, 32], 1, 'Detect', ['nc', 'anchors']],  # Detect(P3, P4, P5, P6)
	 ]
)
with open('./yolov5x6.yaml', 'w') as f:
    yaml.dump(model1_yaml, f, default_flow_style=True)

model2_yaml = dict(
	nc=5,  # number of classes
	depth_multiple=1.33,  # model depth multiple
	width_multiple=1.25,  # layer channel multiple
	anchors=3,  # 这里把默认的anchors配置改成了3以启用autoanchor, 获取针对自己训练时的img_size的更优质的anchor size

	# YOLOv5 backbone
	backbone=
	# [from, number, module, args]
	[[-1, 1, 'Focus', [64, 3]],  # 0-P1/2
	 [-1, 1, 'Conv', [128, 3, 2]],  # 1-P2/4
	 [-1, 3, 'C3', [128]],
	 [-1, 1, 'Conv', [256, 3, 2]],  # 3-P3/8
	 [-1, 9, 'C3', [256]],
	 [-1, 1, 'Conv', [512, 3, 2]],  # 5-P4/16
	 [-1, 9, 'C3', [512]],
	 [-1, 1, 'Conv', [768, 3, 2]],  # 7-P5/32
	 [-1, 3, 'C3', [768]],
	 [-1, 1, 'Conv', [1024, 3, 2]],  # 9-P6/64
	 [-1, 1, 'SPP', [1024, [3, 5, 7]]],
	 [-1, 3, 'C3TR', [1024, 'False']],  # 11  <-------- C3TR() Transformer module
	 ],

	# YOLOv5 head
	head=
	[[-1, 1, 'Conv', [768, 1, 1]],
	 [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
	 [[-1, 8], 1, 'Concat', [1]],  # cat backbone P5
	 [-1, 3, 'C3', [768, 'False']],  # 15

	 [-1, 1, 'Conv', [512, 1, 1]],
	 [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
	 [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
	 [-1, 3, 'C3', [512, 'False']],  # 19

	 [-1, 1, 'Conv', [256, 1, 1]],
	 [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
	 [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
	 [-1, 3, 'C3', [256, 'False']],  # 23 (P3/8-small)

	 [-1, 1, 'Conv', [256, 3, 2]],
	 [[-1, 20], 1, 'Concat', [1]],  # cat head P4
	 [-1, 3, 'C3', [512, 'False']],  # 26 (P4/16-medium)

	 [-1, 1, 'Conv', [512, 3, 2]],
	 [[-1, 16], 1, 'Concat', [1]],  # cat head P5
	 [-1, 3, 'C3', [768, 'False']],  # 29 (P5/32-large)

	 [-1, 1, 'Conv', [768, 3, 2]],
	 [[-1, 12], 1, 'Concat', [1]],  # cat head P6
	 [-1, 3, 'C3', [1024, 'False']],  # 32 (P6/64-xlarge)

	 [[23, 26, 29, 32], 1, 'Detect', ['nc', 'anchors']],  # Detect(P3, P4, P5, P6)
	 ]
)
with open('./yolov5x6-transformer.yaml', 'w') as f:
	yaml.dump(model2_yaml, f, default_flow_style=True)

model3_yaml = dict(
	nc=5,  # number of classes
	depth_multiple=1.0,  # model depth multiple
	width_multiple=1.0,  # layer channel multiple
	anchors=3,  # 这里把默认的anchors配置改成了3以启用autoanchor, 获取针对自己训练时的img_size的更优质的anchor size

	# YOLOv5 backbone
	backbone=
	# [from, number, module, args]
	[[-1, 1, 'Focus', [64, 3]],  # 0-P1/2
	 [-1, 1, 'Conv', [128, 3, 2]],  # 1-P2/4
	 [-1, 3, 'C3', [128]],
	 [-1, 1, 'Conv', [256, 3, 2]],  # 3-P3/8
	 [-1, 9, 'C3', [256]],
	 [-1, 1, 'Conv', [512, 3, 2]],  # 5-P4/16
	 [-1, 9, 'C3TR', [512]],  # <-------- C3TR() Transformer module
	 [-1, 1, 'Conv', [768, 3, 2]],  # 7-P5/32
	 [-1, 3, 'C3', [768]],
	 [-1, 1, 'Conv', [1024, 3, 2]],  # 9-P6/64
	 [-1, 1, 'SPP', [1024, [3, 5, 7]]],
	 [-1, 3, 'C3TR', [1024, 'False']],  # 11  <-------- C3TR() Transformer module
	 ],

	# YOLOv5 head
	head=
	[[-1, 1, 'Conv', [768, 1, 1]],
	 [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
	 [[-1, 8], 1, 'Concat', [1]],  # cat backbone P5
	 [-1, 3, 'C3', [768, 'False']],  # 15

	 [-1, 1, 'Conv', [512, 1, 1]],
	 [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
	 [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
	 [-1, 3, 'C3', [512, 'False']],  # 19

	 [-1, 1, 'Conv', [256, 1, 1]],
	 [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
	 [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
	 [-1, 3, 'C3', [256, 'False']],  # 23 (P3/8-small)

	 [-1, 1, 'Conv', [256, 3, 2]],
	 [[-1, 20], 1, 'Concat', [1]],  # cat head P4
	 [-1, 3, 'C3', [512, 'False']],  # 26 (P4/16-medium)

	 [-1, 1, 'Conv', [512, 3, 2]],
	 [[-1, 16], 1, 'Concat', [1]],  # cat head P5
	 [-1, 3, 'C3', [768, 'False']],  # 29 (P5/32-large)

	 [-1, 1, 'Conv', [768, 3, 2]],
	 [[-1, 12], 1, 'Concat', [1]],  # cat head P6
	 [-1, 3, 'C3', [1024, 'False']],  # 32 (P6/64-xlarge)

	 [[23, 26, 29, 32], 1, 'Detect', ['nc', 'anchors']],  # Detect(P3, P4, P5, P6)
	 ],
)
with open('./yolov5l6-transformer.yaml', 'w') as f:
	yaml.dump(model3_yaml, f, default_flow_style=True)

'''
python ./yolov5/train.py \
          --data /home/dske/workspace/py_stu/limu/cowboy/exam01/my_data_config.yaml \
          --cfg /home/dske/workspace/py_stu/limu/cowboy/exam01/yolov5x6.yaml \
          --hyp /home/dske/workspace/py_stu/limu/cowboy/exam01/my_hyp_config.yaml \
          --cache --exist-ok --multi-scale \
          --project /home/dske/workspace/py_stu/limu/cowboy/exam01/Cow_Boy_Outfits_Detection --name yolov5x6 \
          --img-size 1088 --batch-size 2 --epochs 1 --workers 2 --weights yolov5x6.pt  #  实际train了100个epoch
'''