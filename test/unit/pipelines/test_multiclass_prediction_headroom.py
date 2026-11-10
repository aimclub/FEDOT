from datetime import timedelta

import numpy as np
import pytest

from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.mark.parametrize('classes,task_type,expected', [
    (355, TaskTypesEnum.classification, 65),
    (7, TaskTypesEnum.classification, 90),
    (355, TaskTypesEnum.regression, 90),
])
def test_many_class_predictions_get_headroom(classes, task_type, expected):
    data = InputData(idx=np.arange(classes), features=np.zeros((classes, 2)),
                     target=np.arange(classes), task=Task(task_type), data_type=DataTypesEnum.table)
    pipeline = Pipeline(PipelineNode('lgbm'))
    original = dict(pipeline.root_node.parameters)
    changed = pipeline._set_default_native_fit_time_limits(timedelta(seconds=100), data)
    assert pipeline.root_node.parameters['fit_time_limit'] == expected
    assert changed == [(pipeline.root_node, original)]


def test_explicit_limit_not_replaced():
    data = InputData(idx=np.arange(355), features=np.zeros((355, 2)), target=np.arange(355),
                     task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table)
    node = PipelineNode('lgbm')
    node.parameters = {'fit_time_limit': 95}
    pipeline = Pipeline(node)
    assert pipeline._set_default_native_fit_time_limits(timedelta(seconds=100), data) == []
    assert node.parameters['fit_time_limit'] == 95
