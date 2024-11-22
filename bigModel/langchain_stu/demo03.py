# -*- coding: utf-8 -*-
# @Time : 2024/10/9 20:26
# @Author : nanji
# @Site : 
# @File : demo03.py
# @Software: PyCharm 
# @Comment :

from langchain_community.document_loaders import TextLoader

loader = TextLoader('a.txt', encoding='UTF-8')
data = loader.load()
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    chunk_size=100,  # 切块大小
    chunk_overlap=0,  # 快与块重叠部分
)
texts = text_splitter.split_documents(data)
print(len(texts))
print('1' * 100)
print(type(texts[0]))
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings_model)
# retriever 检索
retriever = db.as_retriever(search_kwargs={'k': 5})
docs = retriever.get_relevant_documents('唐代之科举')
print('2' * 100)
print(docs[1].page_content)
# 网络loader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader('')
data2 = loader.load()
text_splitter2 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter2.split_documents(data2)
print(len(splits))
