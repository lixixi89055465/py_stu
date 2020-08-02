import collections

s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
d = collections.defaultdict(list)
print(list(d.items()))
print(d)
for k, v in s:
    print('a')
    print(k,v)
    d[k].append(v)
    d[k].append(v)

print(s)
print(list(d.items()))

