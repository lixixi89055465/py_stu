def myfun(fun):
    def wrapper(username, password):
        if username == 'root' and password == '123':
            print("登录成功")
        else:
            print("无法登录!")

    return wrapper


@myfun
def fun():
    print("函数运行成功！")


if __name__ == '__main__':
    fun("root", "1234")
