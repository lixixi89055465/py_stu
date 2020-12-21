import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs, sys


def cut_words(sentence):
    return " ".join(jieba.cut(sentence)).encode("utf-8")


f = codecs.open('wiki.zh.jian.text', 'r', encoding='utf-8')
target = codecs.open('zh.jian.wiki.seg-1.3g.txt', 'w', encoding='utf-8')
print('open file')
line_num = 1
line = f.readline()
while (line):
    print('-------- processing    ', line_num, '  article -----------')
    line_seg = " ".join(jieba.cut(line))
    target.writelines(line_seg)
    line_num = line_num + 1
    line = f.readline()
    if line_num > 100:
        break

f.close()
target.close()
exit()
