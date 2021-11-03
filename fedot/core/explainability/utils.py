from typing import Optional
from copy import deepcopy

import fedot.core.composer.metrics as metrics
import fedot.core.pipelines.pipeline as pipeline
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.repository.tasks import TaskTypesEnum


def single_node_pipeline(model: str, custom_params: dict) -> 'Pipeline':
    surrogate_node = PrimaryNode(model)
    surrogate_node.custom_params = custom_params
    return pipeline.Pipeline(surrogate_node)


def fit_naive_surrogate_model(
        black_box_model: 'Pipeline', surrogate_model: 'Pipeline', data: 'InputData',
        metric: 'Metric' = None, **kwargs) -> Optional[float]:

    output_mode = 'default'

    if data.task.task_type == TaskTypesEnum.classification:
        if metric is None:
            metric = metrics.F1
        output_mode = 'labels'

    elif data.task.task_type == TaskTypesEnum.regression:
        if metric is None:
            metric = metrics.R2

    prediction = black_box_model.predict(data, output_mode=output_mode)
    surrogate_model.fit(data, prediction)

    if metric:
        data_c = deepcopy(data)
        data_c.target = surrogate_model.predict(data, output_mode=output_mode).predict
        score = round(abs(metric.metric(data_c, prediction)), 2)

    else:
        score = None

    return score
