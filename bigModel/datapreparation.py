# -*- coding: utf-8 -*-
"""DataPreparation

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zWSZyEadl2b5MuEXeHMn9vJZSPsQ7j-0
"""

import pandas as pd
import datasets
from pprint import pprint
from transformers import AutoTokenizer

tokenizer=AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')

text='Hi,how are you'

encoded_text=tokenizer(text)['input_ids']

encoded_text

decoded_text=tokenizer.decode(encoded_text)
print('Decoded tokens back into text:',decoded_text)

