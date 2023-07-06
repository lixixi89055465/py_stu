import random


def make_code(i):
    res = ""
    for j in range(i):
        mun = str(random.randint(0, 9))
        c = chr(random.randint(65, 90))
        d = chr(random.randint(65, 90)).lower()
        s = random.choice([mun, c, d])
        res += s
    return res


print(make_code(10))