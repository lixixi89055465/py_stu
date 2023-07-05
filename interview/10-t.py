def newrun(func):
    def wrapper(username, password):
        if username == 'root' and password == '123':
            print('通过认证!')
            return func()
        else:
            print('用户名或者密码错误!')
            return

    return wrapper


@newrun
def run():
    print('开始执行函数')


run('root', '123i4')
