from fedot.core.data.data import InputData
from fedot.core.repository.quality_metrics_repository import MetricsRepository


def metric_evaluation(pipeline, train_data: InputData, test_data: InputData,
                      metrics: list, evaluated_metrics: list, vb_number: int = None):
    """ Pipeline training and metrics assessment

    :param pipeline: pipeline for validation
    :param train_data: InputData for train
    :param test_data: InputData for validation
    :param metrics: list with metrics for evaluation
    :param evaluated_metrics: list with metrics values
    :param vb_number: number of validation blocks
    """
    pipeline.fit_from_scratch(train_data)

    for index, metric in enumerate(metrics):
        if callable(metric):
            metric_func = metric
        else:
            metric_func = MetricsRepository().metric_by_id(metric)
        metric_value = metric_func(pipeline, reference_data=test_data, validation_blocks=vb_number)
        evaluated_metrics[index].extend([metric_value])
    return evaluated_metrics
