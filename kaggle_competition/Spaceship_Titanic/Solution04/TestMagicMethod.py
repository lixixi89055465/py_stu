class People(object):
    def __new__(cls, *args, **kwargs):
        print('触发了构造方法')
        ret = super().__new__(cls)  # 调用父类的
        return ret

    def __init__(self, name, age):
        self.name = name
        self.age = age
        print('初始化方法 ')

    def __del__(self):
        print("析构方法 ,删除对象 ")


class MyDict(dict):
    def __getitem__(self, key):
        print('call __getItem__')
        return super().__getitem__(key)

    def __missing__(self, key):
        print('call __missing__ ')
        raise KeyError(f'key {key} not exists')


if __name__ == '__main__':
    dic=MyDict()
    dic['a']=111
    print(dic)
    print('0'*100)
    # print(dic['b'])
    print('1'*100)
    print(dic.get('b', 2323))
    print('2'*100)
    from collections import defaultdict
    mydic=defaultdict(lambda :"未知!")
    mydic.update({'1':"男",'2':"女"})



