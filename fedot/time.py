

class TimeObserver:
    """
    A class whose state changes notify all objects responsible for model training
    (compositing and hyperparameters tuning)
    """

    def __init__(self, time_for_composing, time_for_tuning):
        self.is_first_pipeline_fit = None
        self.is_composing = None
        self.is_tuning = None

    def notify_composing_stopped(self):
        pass

    def notify_tuning_stopped(self):
        pass
