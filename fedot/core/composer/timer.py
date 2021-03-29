import datetime
from abc import ABC
from fedot.core.log import Log, default_log


class Timer(ABC):
    def __init__(self, max_lead_time: datetime.timedelta = None, log: Log = None):
        self.process_terminated = False
        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log
        self.max_lead_time = max_lead_time

    def __enter__(self):
        self.start = datetime.datetime.now()
        return self

    @property
    def start_time(self):
        return self.start

    @property
    def spent_time(self) -> datetime.timedelta:
        return datetime.datetime.now() - self.start

    @property
    def minutes_from_start(self) -> float:
        return self.spent_time.total_seconds() / 60.

    @property
    def seconds_from_start(self) -> float:
        return self.spent_time.total_seconds()

    def is_time_limit_reached(self) -> bool:
        self.process_terminated = False
        if self.max_lead_time is not None:
            if datetime.datetime.now() - self.start >= self.max_lead_time:
                self.process_terminated = True
        return self.process_terminated

    def __exit__(self, *args):
        return self.process_terminated


class CompositionTimer(Timer):
    def __init__(self, max_lead_time: datetime.timedelta = None, log: Log = None):
        super().__init__(max_lead_time=max_lead_time, log=log)
        self.init_time = 0

    def _is_next_iteration_possible(self, time_constraint: float, generation_num: int = None) -> bool:
        minutes = self.minutes_from_start
        if generation_num is not None:
            evo_proc_minutes = minutes - self.init_time
            possible = time_constraint > (minutes + (evo_proc_minutes / (generation_num + 1)))
        else:
            possible = time_constraint > minutes
        if not possible:
            self.process_terminated = True
        return possible

    def is_time_limit_reached(self, generation_num: int = None) -> bool:
        if self.max_lead_time:
            max_lead_time = 0 if self.max_lead_time.total_seconds() < 0 else self.max_lead_time.total_seconds() / 60.
            if max_lead_time:
                reached = not self._is_next_iteration_possible(generation_num=generation_num,
                                                               time_constraint=max_lead_time)
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


class TunerTimer(Timer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_time_limit_reached(self) -> bool:
        super().is_time_limit_reached()
        if self.process_terminated:
            self.log.info('Tuning completed because of the time limit reached')
        return self.process_terminated

    def __exit__(self, *args):
        return self.process_terminated
