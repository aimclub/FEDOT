import datetime
from abc import ABC
from fedot.core.log import Log, default_log


class Timer(ABC):
    def __init__(self, log: Log = None):
        self.process_terminated = False
        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

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

    def is_time_limit_reached(self, max_lead_time: datetime.timedelta) -> bool:
        if datetime.datetime.now() - self.start >= max_lead_time:
            self.process_terminated = True
        else:
            self.process_terminated = False
        return self.process_terminated

    def __exit__(self, *args):
        return self.process_terminated


class CompositionTimer(Timer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _is_next_iteration_possible(self, time_constraint: float, generation_num: int = None,
                                    additional_proc_time: float = None) -> bool:
        minutes = self.minutes_from_start
        if generation_num is not None:
            evo_proc_minutes = minutes
            if additional_proc_time:
                evo_proc_minutes = minutes - additional_proc_time
            possible = time_constraint > (minutes + (evo_proc_minutes / (generation_num + 1)))
        else:
            possible = time_constraint > minutes
        if not possible:
            self.process_terminated = True
        return possible

    def is_time_limit_reached(self, max_lead_time: datetime.timedelta, generation_num: int = None,
                              additional_proc_time: float = None) -> bool:
        max_lead_time = 0 if max_lead_time.total_seconds() < 0 else max_lead_time.total_seconds() / 60.
        if max_lead_time:
            reached = not self._is_next_iteration_possible(generation_num=generation_num, time_constraint=max_lead_time,
                                                           additional_proc_time=additional_proc_time)
        else:
            self.process_terminated = True
            reached = True
        return reached

    def __exit__(self, *args):
        self.log.info(f'Composition time: {round(self.minutes_from_start, 3)} min')
        if self.process_terminated:
            self.log.info('Algorithm was terminated due to processing time limit')


class TunerTimer(Timer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_time_limit_reached(self, max_lead_time: datetime.timedelta) -> bool:
        super().is_time_limit_reached(max_lead_time)
        if self.process_terminated:
            self.log.info('Tuning completed because of the time limit reached')
        return self.process_terminated

    def __exit__(self, *args):
        return self.process_terminated
