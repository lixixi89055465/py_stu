import pandas as pd
import jieba
df_news=pd.read_table("./images/val.txt",names=['category','theme','URL','content'],encoding='utf-8')
df_news=df_news.dropna()

print(df_news.head())