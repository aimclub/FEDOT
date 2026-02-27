import golem.core.log
import numpy as np
import os
import shutil
import pytest
import logging

from fedot_ind.core.architecture.pipelines.abstract_pipeline import AbstractPipeline
from fedot_ind.core.repository.constanst_repository import VALID_LINEAR_REG_PIPELINE, VALID_LINEAR_CLF_PIPELINE, \
    VALID_LINEAR_DETECTION_PIPELINE, VALID_LINEAR_TSF_PIPELINE
from fedot_ind.tools.serialisation.path_lib import PROJECT_PATH


logging.basicConfig(level=logging.INFO)


class LinearPipelineCase:
    def __init__(self, pipeline_label, node_list, data_list, task, task_params=None):
        if task_params is None:
            task_params = {}
        self.pipeline_label = pipeline_label
        self.node_list = node_list
        self.data_list = data_list
        self.task = task
        self.task_params = task_params

    def __str__(self) -> str:
        return self.pipeline_label


def get_data():
    # random train data
    import numpy as np
    train_features = np.random.rand(100, 8)
    train_target = np.zeros(100)

    test_features = np.random.rand(50, 8)
    test_target = np.zeros(50)
    test_target[30:40] = 1.0

    return dict(train_data=(train_features, train_target),
                test_data=(test_features, test_target)
                )


LINEAR_REG_PIPELINE_CASES = [
    LinearPipelineCase(
        pipeline_label=pipeline_label,
        node_list=node_list,
        # data_list=['AppliancesEnergy'],
        data_list=[
            dict(train_data=(np.random.rand(25, 100), np.random.rand(25)),
                 test_data=(np.random.rand(25, 100), np.random.rand(25))),
            dict(train_data=(np.random.rand(25, 3, 100), np.random.rand(25)),
                 test_data=(np.random.rand(25, 3, 100), np.random.rand(25)))
        ],
        task='regression'
    ) for pipeline_label, node_list in VALID_LINEAR_REG_PIPELINE.items()
]

LINEAR_CLF_PIPELINE_CASES = [
    LinearPipelineCase(
        pipeline_label=pipeline_label,
        node_list=node_list,
        data_list=[
            dict(train_data=(np.random.rand(25, 50), np.random.randint(0, 2, 25)),
                 test_data=(np.random.rand(25, 50), np.random.randint(0, 2, 25))),
            dict(train_data=(np.random.rand(25, 3, 50), np.random.randint(0, 2, 25)),
                 test_data=(np.random.rand(25, 3, 50), np.random.randint(0, 2, 25)))
        ],
        # data_list=[
        # 'Earthquakes',
        # 'ERing'
        # ],
        task='classification'
    ) for pipeline_label, node_list in VALID_LINEAR_CLF_PIPELINE.items()
]

LINEAR_TSF_PIPELINE_CASES = [
    LinearPipelineCase(
        pipeline_label=pipeline_label,
        node_list=node_list,
        # data_list=[dict(benchmark='M4', dataset='D2600', task_params={'forecast_length': 14})],
        data_list=[dict(train_data=(np.random.rand(100), np.random.rand(100)),
                        test_data=(np.random.rand(14), np.random.rand(14)),
                        task_params={'forecast_length': 14})],
        task='ts_forecasting',
        task_params=dict(forecast_length=14)
    ) for pipeline_label, node_list in VALID_LINEAR_TSF_PIPELINE.items()
]

LINEAR_DETECTION_PIPELINE_CASES = [
    LinearPipelineCase(
        pipeline_label=pipeline_label,
        node_list=node_list,
        data_list=[get_data()],
        # data_list=[dict(benchmark='valve1', dataset='1')],
        task='classification',
        task_params=dict(industrial_strategy='anomaly_detection', detection_window=10)
    ) for pipeline_label, node_list in VALID_LINEAR_DETECTION_PIPELINE.items()
]

# TODO: temporarily workaround skip some
BANNED_LINEAR_PIPELINE_LABELS = [
    # 'topological_clf',
    # 'topological_reg',
    # 'conv_ae_detector',
    # 'glm'
    'composite_reg',
    'topological_lgbm',
    'composite_clf',
    'stat_detector',
]

LINEAR_PIPELINE_CASES = [case for case in LINEAR_REG_PIPELINE_CASES + LINEAR_CLF_PIPELINE_CASES
                         + LINEAR_DETECTION_PIPELINE_CASES + LINEAR_TSF_PIPELINE_CASES if
                         case.pipeline_label not in BANNED_LINEAR_PIPELINE_LABELS]

# @set_pytest_timeout_in_seconds(300)
# @pytest.mark.xfail()


def mock_message(self, msg: str, **kwargs):
    level = 40
    self.log(level, msg, **kwargs)


@pytest.mark.parametrize('pipeline_case', LINEAR_PIPELINE_CASES, ids=str)
def test_valid_linear_pipelines(pipeline_case: LinearPipelineCase, monkeypatch):
    np.random.seed(34)
    path = os.path.join(PROJECT_PATH, 'cache')

    # to be sure that test run well locally
    if os.path.exists(path):
        shutil.rmtree(path)
    # monkeypatch golem message function
    monkeypatch.setattr(golem.core.log.LoggerAdapter, 'message', mock_message)

    logging.basicConfig(level=logging.INFO)
    if isinstance(pipeline_case.node_list, list):
        pipeline_case.node_list = {0: pipeline_case.node_list}
    result = [
        AbstractPipeline(
            task=pipeline_case.task,
            task_params=pipeline_case.task_params
        ).evaluate_pipeline(pipeline_case.node_list, data) for data in pipeline_case.data_list
    ]
    assert None not in result
