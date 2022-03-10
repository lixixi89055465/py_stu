'''

'''

import csv
import os
from PIL import Image

# train_csv_path = "C:/Users/MMatx/Desktop/研究生/mini-imagenet/mini-imagenet/train.csv"
# val_csv_path = "C:/Users/MMatx/Desktop/研究生/mini-imagenet/mini-imagenet/val.csv"
# test_csv_path = "C:/Users/MMatx/Desktop/研究生/mini-imagenet/mini-imagenet/test.csv"
train_csv_path = "/home/sdb2/aidata/mini-image/train.csv"
val_csv_path = "/home/sdb2/aidata/mini-image/val.csv"
test_csv_path = "/home/sdb2/aidata/mini-image/test.csv"

train_label = {}
val_label = {}
test_label = {}
with open(train_csv_path) as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        train_label[row[0]] = row[1]

with open(val_csv_path) as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        val_label[row[0]] = row[1]

with open(test_csv_path) as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        test_label[row[0]] = row[1]

# img_path = "C:/Users/MMatx/Desktop/研究生/mini-imagenet/mini-imagenet/images"
# new_img_path = "C:/Users/MMatx/Desktop/研究生/mini-imagenet/mini-imagenet/ok"

img_path = "/home/sdb2/aidata/mini-image/images"
new_img_path = "/home/sdb2/aidata/mini-image/images_split"
for png in os.listdir(img_path):
    path = img_path + '/' + png
    im = Image.open(path)
    if (png in train_label.keys()):
        tmp = train_label[png]
        temp_path = new_img_path + '/train' + '/' + tmp
        if (os.path.exists(temp_path) == False):
            os.makedirs(temp_path)
        t = temp_path + '/' + png
        im.save(t)
        # with open(temp_path, 'wb') as f:
        #     f.write(path)

    elif (png in val_label.keys()):
        tmp = val_label[png]
        temp_path = new_img_path + '/val' + '/' + tmp
        if (os.path.exists(temp_path) == False):
            os.makedirs(temp_path)
        t = temp_path + '/' + png
        im.save(t)

    elif (png in test_label.keys()):
        tmp = test_label[png]
        temp_path = new_img_path + '/test' + '/' + tmp
        if (os.path.exists(temp_path) == False):
            os.makedirs(temp_path)
        t = temp_path + '/' + png
        im.save(t)

