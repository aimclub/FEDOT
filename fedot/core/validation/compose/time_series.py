from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from fedot.core.composer.cache import OperationsCache
from fedot.core.data.data import InputData
from fedot.core.validation.compose.metric_estimation import metric_evaluation
from fedot.core.validation.split import ts_cv_generator


def ts_metric_calculation(reference_data: Union[InputData, List[Tuple[InputData, InputData]]],
                          cv_folds: int,
                          validation_blocks: int,
                          metrics: [str, Callable] = None,
                          pipeline=None,
                          cache: Optional[OperationsCache] = None,
                          log=None) -> [Tuple[float, ...], None]:
    """ Determine metric value for time series forecasting pipeline based
    on data for validation

    :param reference_data: InputData for validation
    :param cv_folds: number of folds to split data
    :param validation_blocks: number of validation blocks for time series validation
    :param metrics: name of metric or callable object
    :param pipeline: pipeline for validation
    :param cache: cache manager for fitted models
    :param log: object for logging
    """
    # TODO add support for multiprocessing
    if __name__ != '__main__':
        cache = None

    log.debug(f'Pipeline {pipeline.root_node.descriptive_id} fit for cross validation started')

    try:
        evaluated_metrics = [[] for _ in range(len(metrics))]
        fold_id = 0
        for train_data, test_data, validation_blocks_number in ts_cv_generator(
                reference_data, cv_folds, validation_blocks, log):
            # Calculate metric value for each fold of data
            evaluated_metrics = metric_evaluation(pipeline=pipeline, train_data=train_data,
                                                  test_data=test_data, metrics=metrics,
                                                  evaluated_metrics=evaluated_metrics,
                                                  vb_number=validation_blocks_number,
                                                  fold_id=fold_id, cache=cache)
            fold_id += 1
        evaluated_metrics = tuple(map(lambda x: np.mean(x), evaluated_metrics))
        log.debug(f'Pipeline {pipeline.root_node.descriptive_id} with metrics: {list(evaluated_metrics)}')
    except Exception as ex:
        log.debug(f'{__name__}. Pipeline assessment warning: {ex}. Continue.')
        evaluated_metrics = None
    return evaluated_metrics
