import datetime
from contextlib import contextmanager
from typing import Optional

from fedot.core.constants import COMPOSING_TUNING_PROPORTION, MINIMAL_PIPELINE_NUMBER_FOR_EVALUATION, \
    MINIMAL_SECONDS_FOR_TUNING


class ApiTime:
    """
    A class for performing operations on the AutoML algorithm runtime at the API level

    :param time_for_automl: time for AutoML algorithms in minutes
    """

    def __init__(self, **time_params):
        self.time_for_automl = time_params.get('time_for_automl')
        self.with_tuning = time_params.get('with_tuning')

        self.__define_timeouts_for_stages()

        self.composing_spend_time = datetime.timedelta(minutes=0)

        self.tuning_spend_time = datetime.timedelta(minutes=0)

        self.assumption_fit_spend_time = datetime.timedelta(minutes=0)

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

    def have_time_for_composing(self, pop_size: int, n_jobs: int) -> bool:
        timeout_not_set = self.timedelta_composing is None
        return timeout_not_set or self.assumption_fit_spend_time < self.timedelta_composing * n_jobs / pop_size

    def have_time_for_the_best_quality(self, n_jobs: int):
        timeout_not_set = self.timedelta_automl is None
        if timeout_not_set:
            return True
        return self.assumption_fit_spend_time <= self.timedelta_automl * n_jobs / MINIMAL_PIPELINE_NUMBER_FOR_EVALUATION

    def have_time_for_tuning(self):
        timeout_for_tuning = self.determine_resources_for_tuning()
        return timeout_for_tuning >= MINIMAL_SECONDS_FOR_TUNING

    @contextmanager
    def launch_composing(self):
        """ Wrap composing process with timer """
        starting_time_for_composing = datetime.datetime.now()
        yield
        self.composing_spend_time = datetime.datetime.now() - starting_time_for_composing

    @contextmanager
    def launch_tuning(self):
        """ Wrap tuning process with timer """
        starting_time_for_tuning = datetime.datetime.now()
        yield
        self.tuning_spend_time = datetime.datetime.now() - starting_time_for_tuning

    @contextmanager
    def launch_assumption_fit(self):
        """ Wrap assumption fit process with timer """
        starting_time_for_assumption_fit = datetime.datetime.now()
        yield
        self.assumption_fit_spend_time = datetime.datetime.now() - starting_time_for_assumption_fit

    def determine_resources_for_tuning(self):
        """
        Based on time spend for composing and initial pipeline fit determine
        how much time and how many iterations are needed for tuning

        """
        all_spend_time = self.composing_spend_time + self.assumption_fit_spend_time

        if self.time_for_automl is not None:
            all_timeout = float(self.time_for_automl)
            timeout_in_sec = datetime.timedelta(minutes=all_timeout).total_seconds()
            timeout_for_tuning = timeout_in_sec - all_spend_time.total_seconds()
        else:
            timeout_for_tuning = all_spend_time.total_seconds()
        return timeout_for_tuning

    @property
    def timedelta_composing(self) -> Optional[datetime.timedelta]:
        if self.timeout_for_composing is None:
            return None
        return datetime.timedelta(minutes=self.timeout_for_composing)

    @property
    def timedelta_automl(self) -> Optional[datetime.timedelta]:
        if self.time_for_automl is None:
            return None
        return datetime.timedelta(minutes=self.time_for_automl)
