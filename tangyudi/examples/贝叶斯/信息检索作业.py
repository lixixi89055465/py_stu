import jieba.analyse
from matplotlib import pyplot as plt  # 使用matplot进行画图
import pandas as pd
import numpy as np

content = pd.read_csv('./tmp.data', index_col=False, sep='\t',  # 获取小文档的内容。
                      quoting=3, names=['data'], encoding='utf-8')
df_news = pd.read_csv('./data/val.txt', index_col=False, sep='\t',  # 获取大文档文档的内容。
                      quoting=3, names=['category', 'theme', 'URL', 'content'], encoding='utf-8')
# 获取停用词的内容。
stopwords = pd.read_csv("stopwords.txt", index_col=False, sep="\t", quoting=3, names=['stopword'], encoding='utf-8')
stopwords.head(20)
N = 200  # 设置要参与计算的文档的数目
content.dropna()  # 删除无用的空数据
df_news = df_news.dropna()[:N]
contont_base = content.data[0]  # 使用jieba 分词器进行分词
content_s = jieba.lcut(contont_base)
all_content = df_news.content.values.tolist()
contents_clean = []
all_words = []

for line in all_content:  # 将大文档中的数据进行分词，并放入列表中。同时将停用词筛选，并删除长度小于等于一的词
    line = jieba.lcut(line)
    line_clean = []
    for word in line:
        if word in stopwords or len(word) <= 1:
            continue;
        line_clean.append(word)
        all_words.append(str(word))
    contents_clean.append(line_clean)


def drop_stopwords(contents, stopwords):
    """
    将小文档中的数据进行分词，并放入列表中。同时将停用词筛选，并删除长度小于等于一的词
    :param contents:  要处理的内容
    :param stopwords:  停用词的内容
    :return:  处理后的结果
    """
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

df = pd.DataFrame({'key1': content_1, 'count': [1] * len(content_1)})  # 计算小文档中 获取的数据中的文档的词频
grouped_1 = df.groupby('key1').agg({"count": "sum"}).reset_index()

grouped_1 = grouped_1.sort_values(by=["count"], ascending=[False])  # 根据词频来进行排序
df = pd.DataFrame({'key1': all_words, 'count': [1] * len(all_words)})

all_grouped = df.groupby('key1').agg({"count": "sum"}).reset_index()  # 计算大文档中 获取的数据中的文档的词频
all_grouped = all_grouped.sort_values(by=['count'], ascending=[False])
ni = []
all_grouped.index = all_grouped.key1.values.tolist()
grouped_1.index = grouped_1.key1.values.tolist()

doc_count = []  # 获取小文档中词在大文档中的文档频率
for key in grouped_1.index:
    count = 0
    for single in contents_clean:
        if key in single:
            count += 1
    doc_count.append(count)

grouped_1['doc_count'] = doc_count  # 将文档频率写入分组
all_count = []  # 计算词在大文档中的所有词频
tfi = []  # 计算小文中词 的tfi值
idf = []  # 计算小文中词 的idf值
tfi_c_idf = []  # 计算小文中词 的idf*tfi 值
for key in grouped_1.index:
    icount = all_grouped.loc[key, 'count']
    idoc_count = grouped_1.loc[key, 'doc_count']
    tfi_t = 1 + np.log2(icount)
    tfi.append(tfi_t)
    idf_t = np.log2(N * 1.0 / idoc_count)
    idf.append(idf_t)
    tfi_c_idf.append(tfi_t * idf_t)
    all_count.append(all_grouped.loc[key, 'count'])
grouped_1['all_count'] = all_count
grouped_1['tfi'] = tfi
grouped_1['idf'] = idf
grouped_1['tfi_c_idf'] = tfi_c_idf

grouped_1 = grouped_1.sort_values(by='all_count', ascending=False)
y_tfi = grouped_1.tfi.values.tolist()
y_idf = grouped_1.idf.values.tolist()
y_tfi_c_idf = grouped_1.tfi_c_idf.values.tolist()
# x_3 = [i for i in range(len(grouped_1))]
# x_3 = grouped_1.doc_count.tolist()
x_3 = grouped_1.all_count.tolist()
fig = plt.figure(figsize=(20, 8), dpi=80)  # 设置图形大小

ax1 = fig.add_subplot(2, 1, 1)

ax2 = fig.add_subplot(2, 1, 2)
ax1.set_xscale('log')
ax2.set_xscale('log')
# ax1.invert_xaxis()
# ax2.invert_xaxis()

ax1.scatter(x_3, y_tfi, label="tf", marker='o')  # 使用scatter 方法绘制散点图

ax1.scatter(x_3, y_idf, label="idf", marker='x')

ax2.scatter(x_3, y_tfi_c_idf, label="tfi_c_idf",marker='_')
# plt.xlabel("时间")
# plt.ylabel("地点")

plt.title("tf:O   idf:X   tf*idf: _ ")
# plt.gca().invert_xaxis()
plt.grid(True)
# plt.xscale('symlog')
# python semilogx()
# plt.semilogx()
plt.show()  # 展示
print(x_3[:10])
