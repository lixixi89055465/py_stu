class MySQL:
    __instance = None

    def __init__(self, ip, port):
        self.ip = ip
        self.port = port

    @classmethod
    def from_conf(cls):
        print('类方法')
        if cls.__instance is None:
            print('创建对象!')
            cls.__instance=cls(123,456)
        return cls.__instance


print(MySQL.from_conf())
print(MySQL.from_conf())
print(MySQL.from_conf())
print(MySQL.from_conf())
