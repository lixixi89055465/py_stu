import jieba.analyse
import pandas as pd

content = pd.read_csv('./tmp.data', index_col=False, sep='\t',  #
                      quoting=3, names=['data'], encoding='utf-8')
print(content.data[0])
contont_base = content.data[0]

content_s = jieba.lcut(contont_base)
print(type(content_s))
print("\n")


df=pd.DataFrame({'key1':content_s})
print(df)
grouped=df.groupby('key1')
grouped

