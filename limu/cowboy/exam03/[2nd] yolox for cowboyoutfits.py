# -*- coding: utf-8 -*-
# @Time : 2023/12/27 16:06
# @Author : nanji
# @Site : 
# @File : [2nd] yolox for cowboyoutfits.py
# @Software: PyCharm 
# @Comment :

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter)
# will list all files under the input directory
import os
# for dirname, _, filenames in os.walk('./cowboyoutfits'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
import json
from copy import deepcopy

data_list = ['./cowboyoutfits/train.json']
cat = {87: 1, 1034: 5, 131: 2, 318: 3, 588: 4}

dict_list = []
for idx, data in enumerate(data_list):
	with open(data) as f:
		dict_list.append(json.load(f))

new_data = {}
print(list(dict_list[0].keys()))
new_data['info'] = dict_list[0]['info']
new_categories = []
for category in dict_list[0]['categories']:
	new_category = deepcopy(category)
	new_category['id'] = cat[category['id']]
	new_categories.append(new_category)

new_data['categories'] = new_categories
new_data['annotations'] = []
new_data['image'] = []
print('0' * 100)
print(new_data)
anno_count = 1
anno_id_dict = {}
count = 1
anno_dict = {}
for data in dict_list:
	annotations = []
	for annotation in data['annotations']:
		new_annotation = deepcopy(annotation)
		new_annotation['category_id'] = cat[annotation['category_id']]
		if annotation['image_id'] == anno_count:
			new_annotation['image_id'] = anno_count
			anno_dict[annotation['image_id']] = anno_count
			anno_count += 1
			anno_id_dict[anno_count] = 1
		else:
			new_annotation['image_id'] = anno_dict[annotation['image_id']]
			anno_id_dict[anno_dict[annotation['image_id']]] += 1
		new_annotation['id']=count