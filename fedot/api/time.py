import datetime
from contextlib import contextmanager

from fedot.core.constants import COMPOSING_TUNING_PROPORTION


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
        self.composing_spend_time = datetime.timedelta(minutes=0)

        # Time for tuning
        self.starting_time_for_tuning = None
        self.tuning_spend_time = datetime.timedelta(minutes=0)

    def __define_timeouts_for_stages(self):
        """ Determine timeouts for tuning and composing """
        if self.time_for_automl in [None, -1]:
            self.timeout_for_composing = None
        else:
            # Time for composing based on tuning parameters
            if self.with_tuning:
                self.timeout_for_composing = self.time_for_automl * COMPOSING_TUNING_PROPORTION
            else:
                self.timeout_for_composing = self.time_for_automl

    @contextmanager
    def launch_composing(self):
        """ Wrap composing process with timer """
        self.starting_time_for_composing = datetime.datetime.now()
        yield
        self.composing_spend_time = datetime.datetime.now() - self.starting_time_for_composing

    @contextmanager
    def launch_tuning(self):
        """ Wrap tuning process with timer """
        self.starting_time_for_tuning = datetime.datetime.now()
        yield
        self.tuning_spend_time = datetime.datetime.now() - self.starting_time_for_tuning

    def determine_resources_for_tuning(self, init_pipeline_fit_time: datetime.timedelta):
        """
        Based on time spend for composing and initial pipeline fit determine
        how much time and how many iterations are needed for tuning

        :param init_pipeline_fit_time: time to fit stand-alone pipeline
        """
        all_spended_time = self.composing_spend_time + init_pipeline_fit_time

        if self.time_for_automl is not None:
            all_timeout = float(self.time_for_automl)
            timeout_in_sec = datetime.timedelta(minutes=all_timeout).total_seconds()
            timeout_for_tuning = timeout_in_sec - all_spended_time.total_seconds()
        else:
            timeout_for_tuning = all_spended_time.total_seconds()
        return timeout_for_tuning

    @property
    def datetime_composing(self):
        if self.timeout_for_composing is None:
            return None
        return datetime.timedelta(minutes=self.timeout_for_composing)
