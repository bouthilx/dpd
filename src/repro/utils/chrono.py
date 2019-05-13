import time


class Chrono:
    def __init__(self, *args):
        pass

    def __enter__(self):
        self.start = time.time()
        self.end = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        return self

    @property
    def val(self):
        return self.end - self.start

