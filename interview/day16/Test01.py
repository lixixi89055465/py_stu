import time
from threading import Thread


def target(name='python'):
    for i in range(2):
        print('hello', name)
        time.sleep(1)


# thread 01
thread_01 = Thread(target=target)
thread_01.start()
# thread 02
print("1" * 100)
thread_02 = Thread(target=target, args=('MING',))
thread_02.start()
