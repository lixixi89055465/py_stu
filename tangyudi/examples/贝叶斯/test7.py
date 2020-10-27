import jieba.analyse
import pandas as pd
import numpy as np

content = pd.read_csv('./tmp.data', index_col=False, sep='\t',  #
                      quoting=3, names=['data'], encoding='utf-8')

df_news = pd.read_csv('./data/val.txt', index_col=False, sep='\t',  #
                      quoting=3, names=['category', 'theme', 'URL', 'content'], encoding='utf-8')
stopwords = pd.read_csv("stopwords.txt", index_col=False, sep="\t", quoting=3, names=['stopword'], encoding='utf-8')
stopwords.head(20)
stopwords = stopwords.stopword.values.tolist()
N = 100
content.dropna()
df_news = df_news.dropna()[:N]
print(df_news.shape)
content
# print(content.data[0])
contont_base = content.data[0]
content_s = jieba.lcut(contont_base)
print(type(content_s))
print("\n")

all_content = df_news.content.values.tolist()
#
contents_clean = []
all_words = []
for line in all_content:
    line = jieba.lcut(line)
    line_clean = []
    for word in line:
        if word in stopwords:
            continue;
        line_clean.append(word)
        all_words.append(str(word))
    contents_clean.append(line_clean)
print(all_words[:1])
print(len(all_words))


def drop_stopwords(contents, stopwords):
    contents_clean = []
    all_words = []
    for word in contents:
        if word in stopwords:
            continue
        contents_clean.append(word)
        all_words.append(str(word))
    return contents_clean, all_words


content_1, word_1 = drop_stopwords(content_s, stopwords)
all_content, all_words = drop_stopwords(all_words, stopwords)
print(len(content_1))
print(len(all_content))
print(len(all_words))
#
df = pd.DataFrame({'key1': content_1, 'count': [1] * len(content_1)})
grouped_1 = df.groupby('key1').agg({"count": "sum"}).reset_index()
grouped_1 = grouped_1.sort_values(by=["count"], ascending=[False])
print(grouped_1.head(10))
print(type(grouped_1))

print('1' * 20)
df = pd.DataFrame({'key1': all_words, 'count': [1] * len(all_words)})
#
all_grouped = df.groupby('key1').agg({"count": "sum"}).reset_index()
all_grouped = all_grouped.sort_values(by=['count'], ascending=[False])
print('2' * 10)
print(all_grouped[:10])

print(len(contents_clean))
print(contents_clean[0])
print(all_words[:10])
print('3' * 100)
print("进气" in all_grouped['key1'])

print('4' * 100)
ni = []
# for line in grouped_1.key1.values.tolist():
#     print('line:',line)
#     print(all_grouped.at[line,'count'])
all_grouped.index = all_grouped.key1.values.tolist()
grouped_1.index = grouped_1.key1.values.tolist()
print(all_grouped.loc['万', 'count'])
print(all_grouped.head(10))
print('6' * 100)
doc_count = []
for key in grouped_1.index:
    count = 0
    for single in contents_clean:
        if key in single:
            count += 1
    doc_count.append(count)
grouped_1['doc_count'] = doc_count
print('5' * 100)
all_count = []
tfi = []
idf = []
for key in grouped_1.index:
    icount = all_grouped.loc[key, 'count']
    idoc_count = grouped_1.loc[key, 'doc_count']
    tfi.append(1 + np.log2(icount))
    idf.append(N * 1.0 / idoc_count)
    all_count.append(all_grouped.loc[key, 'count'])
grouped_1['all_count'] = all_count
grouped_1['tfi'] = tfi
grouped_1['idf'] = idf


print(grouped_1.head(10))

# grouped_1['ni']=ni
# print(len(grouped_1))
# print(grouped_1[:20])
#
# print(all_grouped[:10])
#
#
