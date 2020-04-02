import datetime


class CompositionTimer(object):
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.evo_terminated = False

    def __enter__(self):
        self.start = datetime.datetime.now()
        return self

    @property
    def start_time(self):
        return self.start

    @property
    def minutes_from_start(self):
        return (datetime.datetime.now() - self.start).seconds / 60.

    def _is_next_iteration_possible(self, generation_num: int, time_constraint: float) -> bool:
        minutes = self.minutes_from_start
        possible = time_constraint > (minutes + (minutes / (generation_num + 1)))
        if not possible:
            self.evo_terminated = True
        return possible

    def is_max_time_reached(self, max_lead_time: float, generation_num: int) -> bool:
        max_lead_time = 0 if max_lead_time < 0 else max_lead_time
        if max_lead_time:
            reached = not self._is_next_iteration_possible(generation_num, max_lead_time)
        else:
            reached = True
        return reached

    def __exit__(self, *args):
        self.secs = (datetime.datetime.now() - self.start).seconds
        self.minutes = self.secs / 60.
        if self.verbose:
            print(f'Composition time: {round(self.minutes, 3)} min')
            if self.evo_terminated:
                print('Algorithm was terminated due to processing time limit')
