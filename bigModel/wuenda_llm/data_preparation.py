'''
# -*- coding: utf-8 -*-
# @Time : ${DATE} ${TIME}
# @Author : nanji
# @Site : ${SITE}
# @File : ${NAME}.py
# @Software: ${PRODUCT_NAME}
# @Comment :
'''
import pandas as pd
import datasets
from pprint import pprint
import os
from transformers import AutoTokenizer
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用国内hf镜像
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
text='hi,how are you'
encoded_text=tokenizer(text)['input_ids']
print(encoded_text)
print('aaaaaaaa')
