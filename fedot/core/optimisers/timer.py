import datetime
from abc import ABC

from fedot.core.log import default_log


class Timer(ABC):
    def __init__(self, timeout: datetime.timedelta = None):
        self.process_terminated = False
        self.log = default_log(self)
        self.timeout = timeout

    def __enter__(self):
        self.start = datetime.datetime.now()
        return self

    @property
    def start_time(self):
        return self.start

    @property
    def spent_time(self) -> datetime.timedelta:
        return datetime.datetime.now() - self.start_time

    @property
    def minutes_from_start(self) -> float:
        return self.spent_time.total_seconds() / 60.

    @property
    def seconds_from_start(self) -> float:
        return self.spent_time.total_seconds()

    def is_time_limit_reached(self) -> bool:
        self.process_terminated = False
        if self.timeout is not None:
            if datetime.datetime.now() - self.start >= self.timeout:
                self.process_terminated = True
        return self.process_terminated

    def __exit__(self, *args):
        return self.process_terminated


class OptimisationTimer(Timer):
    def __init__(self, timeout: datetime.timedelta = None):
        super().__init__(timeout=timeout)
        self.init_time = 0

    def _is_next_iteration_possible(self, time_constraint: float, iteration_num: int = None) -> bool:
        minutes = self.minutes_from_start
        if iteration_num is not None:
            evo_proc_minutes = minutes - self.init_time
            possible = time_constraint > (minutes + (evo_proc_minutes / (iteration_num + 1)))
        else:
            possible = time_constraint > minutes
        if not possible:
            self.process_terminated = True
        return possible

    def is_time_limit_reached(self, iteration_num: int = None) -> bool:
        if self.timeout:
            timeout = 0 if self.timeout.total_seconds() < 0 else self.timeout.total_seconds() / 60.
            if timeout:
                reached = not self._is_next_iteration_possible(iteration_num=iteration_num,
                                                               time_constraint=timeout)
            else:
                self.process_terminated = True
                reached = True
        else:
            reached = False
        return reached

    def set_init_time(self, init_time: float):
        self.init_time = init_time

    def __exit__(self, *args):
        self.log.info(f'Composition time: {round(self.minutes_from_start, 3)} min')
        if self.process_terminated:
            self.log.info('Algorithm was terminated due to processing time limit')


def get_forever_timer() -> Timer:
    return Timer(timeout=None)
