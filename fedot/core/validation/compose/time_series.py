from typing import Callable, Optional, Tuple
from typing import Union, List

import numpy as np

from fedot.core.composer.cache import OperationsCache
from fedot.core.data.data import InputData
from fedot.core.validation.compose.metric_estimation import metric_evaluation


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
    log.debug(f'Pipeline {pipeline.root_node.descriptive_id} fit for cross validation started')
    try:
        evaluated_metrics = [[] for _ in range(len(metrics))]
        fold_num = 0
        for train_data, test_data, validation_blocks_num in reference_data:
            # Calculate metric value for every fold of data
            evaluated_metrics = metric_evaluation(pipeline=pipeline, train_data=train_data,
                                                  test_data=test_data, metrics=metrics,
                                                  evaluated_metrics=evaluated_metrics,
                                                  vb_number=validation_blocks_num,
                                                  fold_num=fold_num, cache=cache)
            fold_num += 1
        evaluated_metrics = tuple(map(lambda x: np.mean(x), evaluated_metrics))
        log.debug(f'Pipeline {pipeline.root_node.descriptive_id} with metrics: {list(evaluated_metrics)}')
    except Exception as ex:
        log.debug(f'{__name__}. Pipeline assessment warning: {ex}. Continue.')
        evaluated_metrics = None
    return evaluated_metrics
