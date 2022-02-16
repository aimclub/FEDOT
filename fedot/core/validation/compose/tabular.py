from typing import Callable, Optional, Tuple

import numpy as np

from fedot.core.composer.cache import OperationsCache
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.validation.compose.metric_estimation import metric_evaluation
from fedot.core.validation.split import tabular_cv_generator


def table_metric_calculation(reference_data: InputData, cv_folds: int,
                             metrics: [str, Callable], pipeline: Optional[Pipeline],
                             cache: Optional[OperationsCache] = None,
                             log=None) -> [Tuple[float, ...], None]:
    """ Perform cross validation on tabular data for regression and classification tasks

    :param reference_data: InputData for validation
    :param cv_folds: number of folds to split data
    :param metrics: name of metric or callable object
    :param pipeline: Pipeline for validation
    :param cache: cache manager for fitted models
    :param log: object for logging
    """

    log.debug(f'Pipeline {pipeline.root_node.descriptive_id} fit for cross validation started')
    try:
        evaluated_metrics = [[] for _ in range(len(metrics))]
        fold_id = 0
        for train_data, test_data in tabular_cv_generator(reference_data, cv_folds):
            # Calculate metric value for every fold of data
            evaluated_metrics = metric_evaluation(pipeline=pipeline, train_data=train_data,
                                                  test_data=test_data, metrics=metrics,
                                                  evaluated_metrics=evaluated_metrics, fold_id=fold_id,
                                                  cache=cache)
            fold_id += 1
        evaluated_metrics = tuple(map(lambda x: np.mean(x), evaluated_metrics))
        log.debug(f'Pipeline {pipeline.root_node.descriptive_id} with metrics: {list(evaluated_metrics)}')

    except Exception as ex:
        log.debug(f'{__name__}. Pipeline assessment warning: {ex}. Continue.')
        evaluated_metrics = None
    return evaluated_metrics
