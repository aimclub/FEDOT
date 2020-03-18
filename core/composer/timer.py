import time


class Timer(object):
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.evo_terminated = False

    def __enter__(self):
        self.start = time.time()
        return self

    @property
    def start_time(self):
        return self.start

    @property
    def minutes_from_start(self):
        return (time.time() - self.start) / 60.

    def next_iteration_is_possible(self, generation_num, time_constraint) -> bool:
        minutes = self.minutes_from_start
        possible = time_constraint > (minutes + (minutes / (generation_num + 1)))
        if not possible:
            self.evo_terminated = True
        return possible

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.minutes = self.secs / 60.
        if self.verbose:
            print(f'Composition time: {round(self.minutes, 3)} min')
            if self.evo_terminated:
                print(f'Algorithm was terminated due to processing time limit')
