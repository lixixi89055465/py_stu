cutLineFlag = ["？", "！", "。","…",";","."]
a="你好啊，我是笨蛋。你也是笨蛋吧！"
for i,v in enumerate(a):
    if v in cutLineFlag:
        print(i,v)

