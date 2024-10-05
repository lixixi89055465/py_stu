# -*- coding: utf-8 -*-
# @Time : 2024/10/5 21:44
# @Author : nanji
# @Site : 
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
import logging
import time
import torch
import transformers

from bigModel.wuenda_llm.utilities import *
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM

logging = logging.getLogger(__name__)
global_config = None
dataset_name = 'lamini_docs.jsonl'
dataset_path = f"/content/{dataset_name}"
use_hf = False

dataset_path = 'lamini/lamini_docs'
use_hf = True
