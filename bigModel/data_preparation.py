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
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
text='hi,how are you'
encoded_text=tokenizer(text)['input_ids']
print(encoded_text)
