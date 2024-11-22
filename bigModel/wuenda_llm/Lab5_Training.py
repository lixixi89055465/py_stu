# -*- coding: utf-8 -*-
# @Time : 2024/10/5 21:44
# @Author : nanji
# @Site : https://www.bilibili.com/video/BV1kJWUeoEuG/?p=6&spm_id_from=333.880.my_history.page.click&vd_source=50305204d8a1be81f31d861b12d4d5cf
# @File : Lab5_Training.py
# @Software: PyCharm 
# @Comment :
import datasets
import tempfile
import logging
import random
import config
import os
import yaml
import time
import torch
import transformers
import pandas as pd
import jsonlines

from utilities import *
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
# from llama import BasicModelRunner
print("aaaaaaaa")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用国内hf镜像
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logger = logging.getLogger(__name__)
global_config = None

logging = logging.getLogger(__name__)
global_config = None

dataset_name = 'lamini_docs.jsonl'
dataset_path = f"/content/{dataset_name}"
use_hf = False

dataset_path = 'lamini/lamini_docs'
use_hf = True  # 是否使用huggingface
model_name = 'EleutherAI/pythia-70m'
training_config = {
    'model': {
        'pretrained_name': model_name,
        'max_length': 2048
    },
    'datasets': {
        'use_hf': use_hf,
        'path': dataset_path
    },
    'verbose': True
}
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token
# train_dataset,test_dataset=tokenize_and_split_data(training_config,to)


print('cccccccccc')