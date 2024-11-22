# -*- coding: utf-8 -*-
# @Time : 2024/10/7 20:58
# @Author : nanji
# @Site : https://www.bilibili.com/video/BV1zu4y1Z7mc/?p=2&spm_id_from=pageDriver&vd_source=50305204d8a1be81f31d861b12d4d5cf
# @File : model_prompts_and_parsers.py
# @Software: PyCharm 
# @Comment :

import openai
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


def get_completion(prompt, model='gpt-3.5-turbo'):
    messages = [{'role': 'user', 'content': prompt}]
    # openai_stu.C
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message['content']


response = get_completion('what is 1+1')
print(response)

from langchain.chat_models import ChatOpenAI

template_string = '''
Translate the text \ 
that is delimited by triple backticks \
into a style that is {style} . \
text: ```{text}```
'''
from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(template_string)
print('1' * 100)
print(prompt_template.message[0].prompt)
print(prompt_template.message[0].prompt.input_variables)
customer_style='''
American English \
in a calm and repectful tone'''