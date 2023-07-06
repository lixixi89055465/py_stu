from threading import Thread
import time


def task(name):
    print('%s is running ' % name)
    time.sleep(2)
    print('%s is done' % name)


# if __name__ == '__main__':
#     t = Thread(target=task, args=('线程1',))
#     #
#     t.daemon = True
#     t.start()
#     print('主')
from threading import Thread
from multiprocessing import Process
import time


def foo():
    print(123)
    time.sleep(1)
    print('end123')


def bar():
    print(456)
    time.sleep(3)
    print('end 456')


if __name__ == '__main__':
    # 进程
    t1 = Process(target=foo)
    t2 = Process(target=bar)
    t1.daemon = True
    t1.start()
    t2.start()
    print('主')
