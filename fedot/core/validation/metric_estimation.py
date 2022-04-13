import gc
from datetime import timedelta
from typing import List, Iterable, Tuple, Callable, Optional, Sequence, Union

import numpy as np

from fedot.core.composer.cache import OperationsCache
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.quality_metrics_repository import MetricsRepository, MetricsEnum


def metric_evaluation(pipeline: Pipeline,
                      train_data: InputData, test_data: InputData,
                      metrics: list, fold_id: int = None, vb_number: int = None,
                      time_constraint: Optional[timedelta] = None,
                      cache: OperationsCache = None,
                      unfit: bool = False) -> List[float]:
    """ Pipeline training and metrics assessment

    :param pipeline: pipeline for validation
    :param train_data: InputData for train
    :param test_data: InputData for validation
    :param metrics: list with metrics for evaluation
    :param fold_id: id of fold for cross-validation
    :param vb_number: number of validation blocks for time series
    :param time_constraint: optional time constraint for pipeline.fit
    :param cache: instance of cache class
    :param unfit: should we unfit the pipeline
    """
    if cache is not None:
        pipeline.fit_from_cache(cache, fold_id)

    pipeline.fit(train_data, use_fitted=cache is not None, time_constraint=time_constraint)

    if cache is not None:
        cache.save_pipeline(pipeline, fold_id)

    evaluated_metrics = []
    for metric in metrics:
        if callable(metric):
            metric_func = metric
        else:
            metric_func = MetricsRepository().metric_by_id(metric)
        metric_value = metric_func(pipeline, reference_data=test_data, validation_blocks=vb_number)
        evaluated_metrics.append(metric_value)

    if unfit:
        # enforce memory cleaning
        pipeline.unfit()
        gc.collect()

    return evaluated_metrics


def calc_metrics_for_folds(cv_folds_generator: Callable[[], Iterable[Tuple[InputData, InputData]]],
                           pipeline: Pipeline,
                           validation_blocks: int = None,
                           metrics: List[Union[MetricsEnum, Callable]] = None,
                           cache: Optional[OperationsCache] = None,
                           num_of_folds: int = 1,
                           log=None) -> Optional[Sequence[float]]:
    """ Determine metric value for time series forecasting pipeline based
    on data for validation

    :param cv_folds_generator: producer of folds, each fold is a tuple of (train_data, test_data)
    :param pipeline: pipeline for validation
    :param validation_blocks: number of validation blocks, used only for time series validation
    :param metrics: name of metric or callable object
    :param cache: cache manager for fitted models
    :param num_of_folds: number of folds
    :param log: object for logging
    """
    try:
        # TODO add support for multiprocessing
        if __name__ != '__main__':
            cache = None
        log.debug(f'Pipeline {pipeline.root_node.descriptive_id} fit for cross validation started')

        fold_id = 0
        folds_metrics = []
        for train_data, test_data in cv_folds_generator():
            unfit = False
            if fold_id != num_of_folds - 1 and fold_id != 0:
                unfit = True
            evaluated_fold_metrics = metric_evaluation(pipeline, train_data, test_data, metrics,
                                                       vb_number=validation_blocks,
                                                       fold_id=fold_id, cache=cache, unfit=unfit)
            folds_metrics.append(evaluated_fold_metrics)
            fold_id += 1
        folds_metrics = tuple(np.mean(folds_metrics, axis=0))  # averages for each metric over folds
        log.debug(f'Pipeline {pipeline.root_node.descriptive_id} with metrics: {folds_metrics}')
    except Exception as ex:
        log.debug(f'{__name__}. Pipeline assessment warning: {ex}. Continue.')
        folds_metrics = None

    return folds_metrics
