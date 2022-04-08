from fedot.core.operations.model import Model
from fedot.core.repository.quality_metrics_repository import MetricsRepository
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.validation.split import ts_cv_generator, tabular_cv_generator


def collect_intermediate_metric_for_nodes_ts(pipeline,input_data, cv_folds, metric, validation_blocks):
    """
    Function to calculate intermediate metric for all Model nodes for time series forecasting task.
     If node operation is not Model, metric is None

    :param pipeline: pipeline object
    :param input_data: input_data for evaluating
    :param cv_folds: num of cv_folds
    :param metric: metric for evaluation
    :param validation_blocks: num of validation blocks
    """

    test_data, block_number = [(_, test_data, block_number) for _, test_data, block_number in
                               ts_cv_generator(input_data, cv_folds, validation_blocks)][-1][1::]
    for node in pipeline.nodes:
        if isinstance(node.operation, Model):
            if callable(metric):
                metric_func = metric
            else:
                metric_func = MetricsRepository().metric_by_id(metric)
            node.metadata.metric = metric_func(pipeline, reference_data=test_data, validation_blocks=block_number)


def collect_intermediate_metric_for_nodes(pipeline, input_data, cv_folds, metric, validation_blocks=None):
    """
    Function to calculate intermediate metric for all Model nodes. If node operation is not Model, metric is None

    :param pipeline: pipeline object
    :param input_data: input_data for evaluating
    :param cv_folds: num of cv_folds
    :param metric: metric for evaluation
    :param validation_blocks: num of validation blocks (only for time series)
    """
    if input_data.task.task_type == TaskTypesEnum.ts_forecasting:
        collect_intermediate_metric_for_nodes_ts(pipeline, input_data, cv_folds, metric, validation_blocks)
        return

    test_data = [(_, test_data) for _, test_data in
                 tabular_cv_generator(input_data, cv_folds)][-1][1]
    for node in pipeline.nodes:
        if isinstance(node.operation, Model):
            if callable(metric):
                metric_func = metric
            else:
                metric_func = MetricsRepository().metric_by_id(metric)
            node.metadata.metric = metric_func(pipeline, reference_data=test_data)
