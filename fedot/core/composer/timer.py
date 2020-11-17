import datetime
from abc import abstractmethod, ABC


class Timer(ABC):
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.process_terminated = False

    def __enter__(self):
        self.start = datetime.datetime.now()
        return self

    @abstractmethod
    def is_time_limit_reached(self, *args):
        raise NotImplementedError()

    def __exit__(self, *args):
        raise NotImplementedError()


class CompositionTimer(Timer):
    def __init__(self, verbose=True):
        super().__init__(verbose=verbose)

    @property
    def start_time(self):
        return self.start

    @property
    def spent_time(self) -> datetime.timedelta:
        return datetime.datetime.now() - self.start

    @property
    def minutes_from_start(self) -> float:
        return self.spent_time.seconds / 60.

    def _is_next_iteration_possible(self, time_constraint: float, generation_num: int = None) -> bool:
        minutes = self.minutes_from_start
        if generation_num is not None:
            possible = time_constraint > (minutes + (minutes / (generation_num + 1)))
        else:
            possible = time_constraint > minutes
        if not possible:
            self.process_terminated = True
        return possible

    def is_time_limit_reached(self, max_lead_time: datetime.timedelta, generation_num: int = None) -> bool:
        max_lead_time = 0 if max_lead_time.seconds < 0 else max_lead_time.seconds / 60.
        if max_lead_time:
            reached = not self._is_next_iteration_possible(generation_num=generation_num, time_constraint=max_lead_time)
        else:
            self.process_terminated = True
            reached = True
        return reached

    def __exit__(self, *args):
        if self.verbose:
            print(f'Composition time: {round(self.minutes_from_start, 3)} min')
            if self.process_terminated:
                print('Algorithm was terminated due to processing time limit')


class TunerTimer(Timer):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)

    def is_time_limit_reached(self, limit) -> bool:
        if datetime.datetime.now() - self.start >= limit:
            self.process_terminated = True
            if self.verbose:
                print('Tuning completed because of the time limit reached')
        return self.process_terminated

    def __exit__(self, *args):
        return self.process_terminated
