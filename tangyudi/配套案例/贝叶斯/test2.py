import re

def word(text):
    return re.findall('[a-z]+',text.lower())

word("afsf")
