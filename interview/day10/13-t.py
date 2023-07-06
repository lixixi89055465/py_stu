def func(n):
    i = 1
    res = 1
    while i <= n:
        yield res
        i += 1
        res *= i

for i in func(10):
    print(i)
