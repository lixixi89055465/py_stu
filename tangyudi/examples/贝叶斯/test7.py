import jieba.analyse
import pandas as pd
import numpy as np

# 获取小文档的内容。
content = pd.read_csv('./tmp.data', index_col=False, sep='\t',  #
                      quoting=3, names=['data'], encoding='utf-8')
# 获取大文档文档的内容。
df_news = pd.read_csv('./data/val.txt', index_col=False, sep='\t',  #
                      quoting=3, names=['category', 'theme', 'URL', 'content'], encoding='utf-8')
# 获取停用词的内容。
stopwords = pd.read_csv("stopwords.txt", index_col=False, sep="\t", quoting=3, names=['stopword'], encoding='utf-8')
stopwords.head(20)
# stopwords = stopwords.stopword.values.tolist()
# 设置要参与计算的文档的数目
N = 100
# 删除无用的空数据
content.dropna()
df_news = df_news.dropna()[:N]
# print(df_news.shape)
# content
# print(content.data[0])

contont_base = content.data[0]
# 实用jieba 分词器进行分词
content_s = jieba.lcut(contont_base)
# print(type(content_s))
# print("\n")

all_content = df_news.content.values.tolist()
#
contents_clean = []
all_words = []
# 将大文档中的数据进行分词，并放入列表中。同时将停用词筛选，并删除长度小于等于一的词
for line in all_content:
    line = jieba.lcut(line)
    line_clean = []
    for word in line:
        if word in stopwords or len(word) <= 1:
            continue;
        line_clean.append(word)
        all_words.append(str(word))
    contents_clean.append(line_clean)


# print(all_words[:1])
# print(len(all_words))


# 将小文档中的数据进行分词，并放入列表中。同时将停用词筛选，并删除长度小于等于一的词
def drop_stopwords(contents, stopwords):
    contents_clean = []
    all_words = []
    for word in contents:
        if word in stopwords or len(word) <= 1:
            continue
        contents_clean.append(word)
        all_words.append(str(word))
    return contents_clean, all_words


content_1, word_1 = drop_stopwords(content_s, stopwords)
all_content, all_words = drop_stopwords(all_words, stopwords)
# print(len(content_1))
# print(len(all_content))
# print(len(all_words))
# 计算小文档中 获取的数据中的文档的词频
df = pd.DataFrame({'key1': content_1, 'count': [1] * len(content_1)})
grouped_1 = df.groupby('key1').agg({"count": "sum"}).reset_index()
# 根据词频来进行排序
grouped_1 = grouped_1.sort_values(by=["count"], ascending=[False])
df = pd.DataFrame({'key1': all_words, 'count': [1] * len(all_words)})

# print(grouped_1.head(10))
# print(type(grouped_1))

# print('1' * 20)
# 计算大文档中 获取的数据中的文档的词频
all_grouped = df.groupby('key1').agg({"count": "sum"}).reset_index()
all_grouped = all_grouped.sort_values(by=['count'], ascending=[False])
# print('2' * 10)
# print(all_grouped[:10])

# print(len(contents_clean))
# print(contents_clean[0])
# print(all_words[:10])
# print('3' * 100)
# print("进气" in all_grouped['key1'])

# print('4' * 100)
ni = []
# for line in grouped_1.key1.values.tolist():
#     print('line:',line)
#     print(all_grouped.at[line,'count'])
all_grouped.index = all_grouped.key1.values.tolist()
grouped_1.index = grouped_1.key1.values.tolist()
# print(all_grouped.loc['万', 'count'])
# print(all_grouped.head(10))
# print('6' * 100)
# 获取小文档中词在大文档中的文档频率
doc_count = []
for key in grouped_1.index:
    count = 0
    for single in contents_clean:
        if key in single:
            count += 1
    doc_count.append(count)
# 将文档频率写入分组
grouped_1['doc_count'] = doc_count
# print('5' * 100)
all_count = []  # 计算词在大文档中的所有词频
tfi = []  # 计算小文中词 的tfi值
idf = []  # 计算小文中词 的idf值
tfi_c_idf = []  # 计算小文中词 的idf*tfi 值
for key in grouped_1.index:
    icount = all_grouped.loc[key, 'count']
    idoc_count = grouped_1.loc[key, 'doc_count']
    tfi_t = 1 + np.log2(icount)
    tfi.append(tfi_t)
    idf_t = N * 1.0 / idoc_count
    idf.append(idf_t)
    tfi_c_idf.append(tfi_t * idf_t)
    all_count.append(all_grouped.loc[key, 'count'])
grouped_1['all_count'] = all_count
grouped_1['tfi'] = tfi
grouped_1['idf'] = idf
grouped_1['tfi_c_idf'] = tfi_c_idf
# print(grouped_1.head(10))

# 实用matplot进行画图
from matplotlib import pyplot as plt

grouped_1.sort_values(by='doc_count', ascending=True)
y_tfi = grouped_1.tfi.values.tolist()
y_idf = grouped_1.idf.values.tolist()
y_tfi_c_idf = grouped_1.tfi_c_idf.values.tolist()
x_3 = grouped_1.all_count.tolist()
# 设置图形大小
fig = plt.figure(figsize=(20, 8), dpi=80)

ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

# 使用scatter 方法绘制散点图
ax1.scatter(x_3, y_tfi, label="tfi")
ax1.scatter(x_3, y_idf, label="idf")
ax2.scatter(x_3, y_tfi_c_idf, label="tfi_c_idf")
# plt.xscale('symlog')
plt.xscale('log')

plt.grid(True)

# 展示
plt.show()
