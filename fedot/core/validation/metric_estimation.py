from datetime import timedelta
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from fedot.core.composer.cache import OperationsCache
from fedot.core.data.data import InputData
from fedot.core.log import Log
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.quality_metrics_repository import MetricsRepository, MetricType


DataSource = Callable[[], Iterable[Tuple[InputData, InputData]]]


def fit_and_eval_metrics(pipeline: Pipeline,
                         train_data: InputData,
                         test_data: InputData,
                         metrics: Sequence[MetricType],
                         fold_id: Optional[int] = None,
                         vb_number: Optional[int] = None,
                         time_constraint: Optional[timedelta] = None,
                         cache: Optional[OperationsCache] = None) -> Sequence[float]:
    """ Pipeline training and metrics assessment

    :param pipeline: pipeline for validation
    :param train_data: InputData for train
    :param test_data: InputData for validation
    :param metrics: metrics for evaluation
    :param fold_id: id of fold for cross-validation
    :param vb_number: number of validation blocks for time series
    :param time_constraint: optional time constraint for pipeline.fit
    :param cache: instance of cache class
    """
    pipeline.fit(
        train_data,
        use_fitted=pipeline.fit_from_cache(cache, fold_id),
        time_constraint=time_constraint
    )

    if cache is not None:
        cache.save_pipeline(pipeline, fold_id)

    return eval_metrics(metrics, test_data, vb_number, pipeline)


def eval_metrics(metrics: Sequence[MetricType],
                 reference_data: InputData,
                 validation_blocks: Optional[int],
                 pipeline: Pipeline) -> Sequence[float]:
    evaluated_metrics = []
    for metric in metrics:
        metric_func = MetricsRepository().metric_by_id(metric, default_callable=metric)
        metric_value = metric_func(pipeline, reference_data=reference_data, validation_blocks=validation_blocks)
        evaluated_metrics.append(metric_value)
    return evaluated_metrics


def calc_metrics_for_folds(cv_folds_generator: DataSource,
                           pipeline: Pipeline,
                           metrics: Sequence[MetricType],
                           validation_blocks: Optional[int] = None,
                           cache: Optional[OperationsCache] = None,
                           log: Log = None) -> Optional[Sequence[float]]:
    """ Determine metric value for time series forecasting pipeline based
    on data for validation

    :param cv_folds_generator: producer of folds, each fold is a tuple of (train_data, test_data)
    :param pipeline: pipeline for validation
    :param validation_blocks: number of validation blocks, used only for time series validation
    :param metrics: name of metric or callable object
    :param cache: cache manager for fitted models
    :param log: object for logging
    """
    log.debug(f'Pipeline {pipeline.root_node.descriptive_id} fit for cross validation started')
    try:
        fold_id = 0
        folds_metrics = []
        for train_data, test_data in cv_folds_generator():
            evaluated_fold_metrics = fit_and_eval_metrics(pipeline, train_data, test_data, metrics,
                                                          vb_number=validation_blocks,
                                                          fold_id=fold_id, cache=cache)
            folds_metrics.append(evaluated_fold_metrics)
            fold_id += 1
        folds_metrics = tuple(np.mean(folds_metrics, axis=0))  # averages for each metric over folds
        log.debug(f'Pipeline {pipeline.root_node.descriptive_id} with metrics: {folds_metrics}')
    except Exception as ex:
        log.debug(f'{__name__}. Pipeline assessment warning: {ex}. Continue.')
        folds_metrics = None
    return folds_metrics
