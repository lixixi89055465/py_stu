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
