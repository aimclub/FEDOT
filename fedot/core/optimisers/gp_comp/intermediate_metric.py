from copy import deepcopy

from fedot.core.operations.model import Model
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.quality_metrics_repository import MetricsRepository


def collect_intermediate_metric_for_nodes(pipeline, input_data, metric, validation_blocks=None):
    for node in pipeline.nodes:
        if isinstance(node.operation, Model):
            if callable(metric):
                metric_func = metric
            else:
                metric_func = MetricsRepository().metric_by_id(metric)
            new_pipeline = Pipeline(node)
            new_pipeline.preprocessor = pipeline.preprocessor

            node.metadata.metric = metric_func(new_pipeline, reference_data=input_data,
                                               validation_blocks=validation_blocks)


def collect_intermediate_metric_for_nodes_cv(pipeline, cv_generator, metric, validation_blocks=None):
    """
    Function to calculate intermediate metric for all Model nodes. If node operation is not Model, metric is None

    :param pipeline: pipeline object
    :param cv_generator: cv generator for folds
    :param metric: metric for evaluation
    :param validation_blocks: num of validation blocks (only for time series)
    """
    test_data = [test_data for _, test_data in
                 cv_generator()][-1]
    collect_intermediate_metric_for_nodes(pipeline, test_data, metric, validation_blocks=validation_blocks)
