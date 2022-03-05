import datetime

from fedot.pattern_wrappers import singleton

COMPOSING_TUNING_PROPORTION = 0.6


@singleton
class ApiTime:
    """
    A class for performing operations on the AutoML algorithm runtime at the API level

    :param time_for_automl: time for AutoML algorithms in minutes
    """

    def __init__(self, **time_params):
        self.time_for_automl = time_params.get('time_for_automl')
        self.with_tuning = time_params.get('with_tuning')

        self.__define_timeouts_for_stages()

        # Time for composing
        self.starting_time_for_composing = None
        self.composing_spend_time = 0

        # Time for tuning
        self.starting_time_for_tuning = None
        self.tuning_spend_time = 0

        # Time to fit stand-alone pipeline
        self.init_pipeline_fit_time = None

    def __define_timeouts_for_stages(self):
        """ Determine timeouts for tuning and composing """
        if self.time_for_automl is None:
            self.timeout_for_composing = None
        else:
            # Time for composing based on tuning parameters
            if self.with_tuning:
                self.timeout_for_composing = self.time_for_automl * COMPOSING_TUNING_PROPORTION
            else:
                self.timeout_for_composing = self.time_for_automl

    def start_composing(self):
        self.starting_time_for_composing = datetime.datetime.now()

    def end_composing(self):
        if self.starting_time_for_composing is None:
            # Composing has not been started previously
            return self.composing_spend_time

        self.composing_spend_time = datetime.datetime.now() - self.starting_time_for_composing
        return self.composing_spend_time

    def start_tuning(self):
        self.starting_time_for_tuning = datetime.datetime.now()

    def end_tuning(self):
        if self.starting_time_for_tuning is None:
            # Pipeline tuning has not been started previously
            return self.tuning_spend_time
        self.tuning_spend_time = datetime.datetime.now() - self.starting_time_for_tuning
        return self.tuning_spend_time

    def determine_resources_for_tuning(self):
        """
        Based on time spend for composing and initial pipeline fit determine
        how much time and how many iterations are needed for tuning
        """
        spended_time_for_composing = self.end_composing() + self.init_pipeline_fit_time
        all_timeout = float(self.time_for_automl)

        iterations = 20 if all_timeout is None else 1000
        if self.time_for_automl not in [None, -1]:
            timeout_in_sec = datetime.timedelta(minutes=all_timeout).total_seconds()
            timeout_for_tuning = timeout_in_sec - spended_time_for_composing.total_seconds()
        else:
            timeout_for_tuning = spended_time_for_composing.total_seconds()
        return timeout_for_tuning, iterations

    @property
    def datetime_composing(self):
        return datetime.timedelta(minutes=self.timeout_for_composing)
