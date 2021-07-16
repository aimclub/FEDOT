from typing import Callable, Tuple, Optional

import numpy as np

from fedot.core.data.data import InputData
from fedot.core.validation.split import ts_cv_generator
from fedot.core.validation.compose.metric_estimation import metric_evaluation
from fedot.core.log import default_log


def ts_metric_calculation(reference_data: InputData, cv_folds: int = 3,
                          metrics: [str, Callable] = None,
                          pipeline: Optional = None, log=None) -> Tuple[float, ...]:
    """ Determine metric value for time series forecasting pipeline based
    on data for validation

    :param reference_data: InputData for validation
    :param cv_folds: number of folds to split data
    :param metrics: name of metric or callable object
    :param pipeline: Pipeline for validation
    :param log: object for logging
    """
    if not log:
        log = default_log(__name__)

    log.debug(f'Pipeline {pipeline.root_node.descriptive_id} fit for cross validation started')
    try:
        evaluated_metrics = [[] for _ in range(len(metrics))]
        for train_data, test_data in ts_cv_generator(reference_data, cv_folds, log):
            # Update information about metrics
            evaluated_metrics = metric_evaluation(pipeline, train_data,
                                                  test_data, metrics,
                                                  evaluated_metrics)
        evaluated_metrics = tuple(map(lambda x: np.mean(x), evaluated_metrics))
    except Exception as ex:
        log.info(f'Pipeline assessment warning: {ex}. Continue.')
        evaluated_metrics = None

    evaluated_metrics = tuple(map(lambda x: np.mean(x), evaluated_metrics))
    return evaluated_metrics
