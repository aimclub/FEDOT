import logging
import os
import shutil

import pytest

from fedot.api.main import Fedot
from fedot.core.repository.tasks import TsForecastingParams
from fedot.core.utils import fedot_project_root
from fedot.remote.infrastructure.clients.test_client import TestClient
from fedot.remote.remote_evaluator import RemoteEvaluator, RemoteTaskParams


@pytest.fixture(autouse=True)
def run_around_tests():
    yield
    # return evaluator to local mode
    evaluator = RemoteEvaluator()
    evaluator.init(None, RemoteTaskParams(mode='local'))


def test_pseudo_remote_composer_classification():
    connect_params = {}
    exec_params = {
        'container_input_path': os.path.join(fedot_project_root(), 'test', 'data'),
        'container_output_path': os.path.join(fedot_project_root(), 'test', 'data', 'remote'),
        'container_config_path': os.path.join(fedot_project_root(), 'test', 'data', '.'),
        'container_image': "test",
        'timeout': 1
    }

    remote_task_params = RemoteTaskParams(
        mode='remote',
        dataset_name='advanced_classification')

    client = TestClient(connect_params, exec_params,
                        output_path=os.path.join(fedot_project_root(), 'test', 'data', 'remote'))

    evaluator = RemoteEvaluator()

    evaluator.init(
        client=client,
        remote_task_params=remote_task_params
    )

    composer_params = {
        'pop_size': 3,
        'cv_folds': None,
        'with_tuning': False,
        'preset': 'best_quality',
        'show_progress': False
    }

    automl = Fedot(problem='classification', timeout=0.1, **composer_params)

    path = os.path.join(fedot_project_root(), 'test', 'data', 'advanced_classification.csv')

    automl.fit(path)
    predict = automl.predict(path)
    shutil.rmtree(os.path.join(fedot_project_root(), 'test', 'data', 'remote', 'fitted_pipeline'))  # recursive deleting

    assert predict is not None


def test_pseudo_remote_composer_ts_forecasting():
    connect_params = {}
    exec_params = {
        'container_input_path': os.path.join(fedot_project_root(), 'test', 'data'),
        'container_output_path': os.path.join(fedot_project_root(), 'test', 'data', 'remote'),
        'container_config_path': os.path.join(fedot_project_root(), 'test', 'data', '.'),
        'container_image': "test",
        'timeout': 1
    }

    remote_task_params = RemoteTaskParams(
        mode='remote',
        dataset_name='short_time_series')

    client = TestClient(connect_params, exec_params,
                        output_path=os.path.join(fedot_project_root(), 'test', 'data', 'remote'))

    evaluator = RemoteEvaluator()

    evaluator.init(
        client=client,
        remote_task_params=remote_task_params
    )

    composer_params = {
        'pop_size': 10,
        'cv_folds': None,
        'with_tuning': False,
        'show_progress': False
    }

    preset = 'best_quality'
    automl = Fedot(problem='ts_forecasting', timeout=0.2, task_params=TsForecastingParams(forecast_length=1),
                   preset=preset, **composer_params)

    path = os.path.join(fedot_project_root(), 'test', 'data', 'short_time_series.csv')

    automl.fit(path, target='sea_height')
    predict = automl.predict(path)
    shutil.rmtree(os.path.join(fedot_project_root(), 'test', 'data', 'remote', 'fitted_pipeline'))  # recursive deleting
    assert predict is not None
