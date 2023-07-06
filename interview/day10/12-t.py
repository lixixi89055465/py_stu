try:
    file = open('text.txt', 'r')
    file.read()
    print(file)
except Exception as e:
    file = open("test.txt", 'w',encoding='utf-8')
    file.write('好好学习，天天向上')
finally:
    file.close()
