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





os.makedirs('./data/images/train', exist_ok=True)
os.makedirs('./data/images/val', exist_ok=True)

train_img = []
# Move train images
for j in data['images']:
    for i in train_id:
        if j['id'] == i:
            shutil.copy('../input/cowboyoutfits/images/' + j['file_name'], './data/images/train')
            train_img.append(j)

val_img = []
# Move val images
for j in data['images']:
    for i in val_id:
        if j['id'] == i:
            shutil.copy('../input/cowboyoutfits/images/' + j['file_name'], './data/images/val')
            val_img.append(j)

len(val_img), len(train_img)