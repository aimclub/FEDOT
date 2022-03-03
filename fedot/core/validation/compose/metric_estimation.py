import gc

from fedot.core.composer.cache import OperationsCache
from fedot.core.data.data import InputData
from fedot.core.repository.quality_metrics_repository import MetricsRepository


def metric_evaluation(pipeline, train_data: InputData, test_data: InputData,
                      metrics: list, evaluated_metrics: list,
                      fold_id: int = None, vb_number: int = None,
                      cache: OperationsCache = None):
    """ Pipeline training and metrics assessment

    :param pipeline: pipeline for validation
    :param train_data: InputData for train
    :param test_data: InputData for validation
    :param metrics: list with metrics for evaluation
    :param evaluated_metrics: list with metrics values
    :param fold_id: id of fold for cross-validation
    :param vb_number: number of validation blocks for time series
    :param cache: instance of cache class
    """
    if cache is not None:
        pipeline.fit_from_cache(cache, fold_id)

    pipeline.fit(train_data, use_fitted=cache is not None)

    for index, metric in enumerate(metrics):
        if callable(metric):
            metric_func = metric
        else:
            metric_func = MetricsRepository().metric_by_id(metric)
        metric_value = metric_func(pipeline, reference_data=test_data, validation_blocks=vb_number)
        evaluated_metrics[index].extend([metric_value])

    if cache is not None:
        cache.save_pipeline(pipeline, fold_id)

    # enforce memory cleaning
    pipeline.unfit()
    gc.collect()

    return evaluated_metrics
