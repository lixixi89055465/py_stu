# -*- coding: utf-8 -*-
# @Time : 2024/10/7 15:01
# @Author : nanji
# @Site : https://www.bilibili.com/video/BV1kJWUeoEuG/?p=7&spm_id_from=pageDriver&vd_source=50305204d8a1be81f31d861b12d4d5cf
# @File : Lab6_Evaluation.py
# @Software: PyCharm
# @Comment :

import datasets
import tempfile
import logging
import random
import config
import os
import yaml
import difflib
import pandas as pd
import transformers
import torch

from tqdm import tqdm
from utilities import *
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用国内hf镜像
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
print(os.environ["HF_ENDPOINT"])
logger = logging.getLogger(__name__)
global_config = None

dataset = datasets.load_dataset('lamini/lamini_docs')
test_dataset = dataset['test']
print(test_dataset[0]['question'])
print(test_dataset[0]['answer'])

model_name = 'lamini/lamini_docs_finetuned'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def is_exact_match(a, b):
    return a.strip() == b.strip()


def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
    # Tokenize
    tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer.encode(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=max_input_tokens
    )
    # Generate
    device = model.device
    generated_tokens_with_prompt = model.generate(
        input_ids=input_ids.to(device),
        max_length=max_output_tokens
    )
    # Decode
    generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)
    # Strip the prompt
    generated_text_answer = generated_text_with_prompt[0][len(text):]
    return generated_text_answer


model.eval()
test_question = test_dataset[0]['question']
generated_answer = inference(test_question, model, tokenizer)
print(test_question)
print(generated_answer)
print('1' * 100)
answer = test_dataset[0]['answer']
print(answer)

exact_match = is_exact_match(generated_answer, answer)
print('2' * 100)
print(exact_match)
n = 10
metrics = {'exact_matches': []}
predictions = []
for i, item in tqdm(enumerate(test_dataset)):
    print('I Evaluating: ' + str(item))
    question = item['question']
    answer = item['answer']
    try:
        predicted_answer = inference(question, model, tokenizer)
    except:
        continue
    predictions.append([predicted_answer, answer])
    exact_match = is_exact_match(generated_answer, answer)
    metrics['exact_matches'].append(exact_match)
    if i > n and n != -1:
        break
print('Number of exact matches:', sum(metrics['exact_matches']))

df = pd.DataFrame(predictions, columns=['predicted_answer', 'target_answer'])
print(df)

evaluation_dataset_path = 'lamini/lamini_docs_evaluation'

evaluation_dataset = datasets.load_dataset(evaluation_dataset_path)
r1 = pd.DataFrame(evaluation_dataset)
print(r1)
# python lm-evaluation-harness/main.py --model hf-causal --model_args pretrained=lamini/lamini_docs_finetuned --tasks arc_easy --device cpu