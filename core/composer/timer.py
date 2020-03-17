import time


class Timer(object):
    def __init__(self, verbose=True):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    @property
    def start_time(self):
        return self.start

    @property
    def minutes_from_start(self):
        return (time.time() - self.start) / 60.

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        self.minutes = self.secs / 60.
        if self.verbose:
            print(f'composition time: {self.minutes} min')
