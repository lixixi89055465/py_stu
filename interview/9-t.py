import os


def fun(path, deep):
    if deep <= 0:
        return
    list_path = os.listdir(path)
    # 把所有的文件和文件夹展示再列表里面
    for i in list_path:
        full_path = os.path.join(path, i)
        if os.path.isdir(full_path):
            fun(full_path, deep - 1)
        else:
            print(full_path)


path = r'D:/'

fun(path, 2)
