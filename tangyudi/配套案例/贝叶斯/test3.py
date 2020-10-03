# %
# from collections import defaultdict
# s=[('yellow',1),('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
# d=defaultdict(list)
# for k, v in s:
#     d[k].append(v)
# a=sorted(d.items())
# print(a)

from collections import defaultdict

s = [('yellow', 1), ('blue', 2), ('yello', 3), ('blue', 4), ('red', 1), ('yello', 333)]
print(type(s[0]))
print(type(s))
d = defaultdict(list)
print(type(d))

for k, v in s:
    d[k].append(v)
s = sorted(d.items())
print(s)
print(d)

