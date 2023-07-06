class Foo:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __call__(self, *args, **kwargs):
        print(args, kwargs)


obj = Foo(1, 2)
print(obj())
print("1" * 100)
print(obj(1, 2, a=3, b=4))

print("2" * 100)
obj.__call__(obj,1,2,a=3,b=4)
