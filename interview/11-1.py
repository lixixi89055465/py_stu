with open('111.xml','rb') as file:
    a=file.read()

with open('new.png','wb') as file:
    file.write(a)