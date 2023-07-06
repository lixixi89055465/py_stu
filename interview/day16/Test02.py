import time
from threading import Thread


class MyThread(Thread):
    def __init__(self, type="Python"):
        super().__init__()
        self.type = type

    def run(self):
        for i in range(2):
            print('hello', self.type)
            time.sleep(1)


if __name__ == '__main__':
    thread_01 = MyThread()
    thread_02 = MyThread("MING")
    thread_01.start()
    thread_02.start()

