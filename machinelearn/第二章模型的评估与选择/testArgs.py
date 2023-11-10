def fun(*args, **kwargs):
    print('args=', args)
    print('kwargs=', kwargs)


fun(1, 2, 3, 4, A='a', B='b', C='c', D='d')
args = (1, 2, 3, 4)
kwargs = {'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd'}

print('0' * 100)


def fun(name, *args):
    print('你好:', name)
    for i in args:
        print("你的宠物有:", i)


fun("Geek", "dog", "cat")
print('1' * 100)


def fun(**kwargs):
    for key, value in kwargs.items():
        print("{0} 喜欢 {1}".format(key, value))


fun(Geek="cat", cat="box")
print('3' * 100)


def fun(data1, data2, data3):
    print("data1: ", data1)
    print("data2: ", data2)
    print("data3: ", data3)


args = ("one", 2, 3)
fun(*args)
kwargs = {"data3": "one", "data2": 2, "data1": 3}
fun(**kwargs)
a, b, *c = 0, 1, 2, 3
print(a)
print(b)
print(c)
