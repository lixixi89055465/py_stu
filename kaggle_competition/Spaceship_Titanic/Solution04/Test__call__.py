class Fib(object):
    def __init__(self):
        self.aa = 12
        pass

    def __lt__(self, other):
        return self.aa > other.aa

    def __le__(self, other):
        return self.aa<other.aa
    def __eq__(self, other):
        return self.aa>other.aa

    def __call__(self, num):
        a, b = 0, 1
        self.l = []
        for i in range(num):
            self.l.append(a)
            a, b = b, a + b
        return self.l

    def __str__(self):
        return str(self.l)

    def __len__(self):
        return len(self.l)

    def __bool__(self):
        return True

    def __delattr__(self, item):
        print('bbbbb')

    __repr__ = __str__


f = Fib()
print(f(10))
print('0' * 100)
print(repr(f))

print('1' * 100)
print(len(f))
print('2' * 100)
print(bool(f))

print('3' * 100)
print(getattr(f, 'aa'))

print('4' * 100)
setattr(f, 'aa',23)
print(getattr(f, 'aa'))
print('5' * 100)
delattr(f, 'aa')
print(getattr(f, 'aa'))
g=Fib()
print('6'*100)
print(f == g)
