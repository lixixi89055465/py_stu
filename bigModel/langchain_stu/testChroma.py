# -*- coding: utf-8 -*-
# @Time : 2024/10/9 21:46
# @Author : nanji
# @Site : 
# @File : testChroma.py
# @Software: PyCharm 
# @Comment :https://blog.csdn.net/qqxx6661/article/details/134289730

import chromadb
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="my_collection")

collection.add(
    documents=["This is a document about engineer", "This is a document about steak"],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}],
    ids=["id1", "id2"]
)

results = collection.query(
    query_texts=["Which food is the best?"],
    n_results=1
)

print(results)
