import codecs, sys

f = codecs.open('wiki.zh.jian.text', 'r', encoding='utf8')
line = f.readline()
print(line)
