from typing import Callable, Tuple, Optional, List, Union

import numpy as np

from fedot.core.composer.cache import OperationsCache
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.validation.compose.metric_estimation import metric_evaluation


def table_metric_calculation(reference_data: Union[InputData, List[Tuple[InputData, InputData]]],
                             metrics: [str, Callable], pipeline: Optional[Pipeline],
                             cache: Optional[OperationsCache] = None,
                             log=None) -> [Tuple[float, ...], None]:
    """ Perform cross validation on tabular data for regression and classification tasks

    :param reference_data: InputData for validation
    :param metrics: name of metric or callable object
    :param pipeline: Pipeline for validation
    :param cache: cache manager for fitted models
    :param log: object for logging
    """
    if ((isinstance(reference_data, InputData) and reference_data.task.task_type is TaskTypesEnum.clustering) or
            (isinstance(reference_data, List) and reference_data[0][0].task.task_type is TaskTypesEnum.clustering)):
        raise NotImplementedError(f"Tabular cross validation for clustering is not supported")

    log.debug(f'Pipeline {pipeline.root_node.descriptive_id} fit for cross validation started')
    try:
        evaluated_metrics = [[] for _ in range(len(metrics))]
        for fold_num, data_pair in enumerate(reference_data):
            train_data, test_data = data_pair
            # Calculate metric value for every fold of data
            evaluated_metrics = metric_evaluation(pipeline=pipeline, train_data=train_data,
                                                  test_data=test_data, metrics=metrics,
                                                  evaluated_metrics=evaluated_metrics, fold_num=fold_num, cache=cache)
        evaluated_metrics = tuple(map(lambda x: np.mean(x), evaluated_metrics))
        log.debug(f'Pipeline {pipeline.root_node.descriptive_id} with metrics: {list(evaluated_metrics)}')

    except Exception as ex:
        log.debug(f'{__name__}. Pipeline assessment warning: {ex}. Continue.')
        evaluated_metrics = None
    return evaluated_metrics
