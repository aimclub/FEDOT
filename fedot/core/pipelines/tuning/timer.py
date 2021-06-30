from fedot.core.optimisers.timer import Timer


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
