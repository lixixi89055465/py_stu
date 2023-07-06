with open("new.png", 'rb') as file:
    a = file.read()

with open('111.xml', 'wb') as c:
    c.write(a)
