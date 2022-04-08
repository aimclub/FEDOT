from fedot.core.operations.model import Model
from fedot.core.repository.quality_metrics_repository import MetricsRepository
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.validation.split import ts_cv_generator, tabular_cv_generator


def collect_intermediate_metric_for_nodes_ts(pipeline, args):
    input_data = args[0]
    num_of_folds = args[1]
    num_of_validation_blocks = args[2]
    test_data, block_number = [(_, test_data, block_number) for _, test_data, block_number in
                               ts_cv_generator(input_data, num_of_folds, num_of_validation_blocks)][-1][1::]
    metric_to_use = args[3][0]
    for node in pipeline.nodes:
        if isinstance(node.operation, Model):
            if callable(metric_to_use):
                metric_func = metric_to_use
            else:
                metric_func = MetricsRepository().metric_by_id(metric_to_use)
            node.metadata.metric = metric_func(pipeline, reference_data=test_data, validation_blocks=block_number)


def collect_intermediate_metric_for_nodes(pipeline, args):
    input_data = args[0]
    num_of_folds = args[1]
    if input_data.task.task_type == TaskTypesEnum.ts_forecasting:
        collect_intermediate_metric_for_nodes_ts(pipeline, args)
        return
    metric_to_use = args[2][0]
    test_data = [(_, test_data) for _, test_data in
                 tabular_cv_generator(input_data, num_of_folds)][-1][1]
    for node in pipeline.nodes:
        if isinstance(node.operation, Model):
            if callable(metric_to_use):
                metric_func = metric_to_use
            else:
                metric_func = MetricsRepository().metric_by_id(metric_to_use)
            node.metadata.metric = metric_func(pipeline, reference_data=test_data)