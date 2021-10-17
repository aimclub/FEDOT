import os

from fedot.api.main import Fedot
from fedot.core.repository.tasks import TsForecastingParams
from fedot.core.utils import fedot_project_root
from fedot.remote.infrastructure.clients.test_client import TestClient
from fedot.remote.remote_evaluator import RemoteEvaluator, RemoteTaskParams
from fedot.remote.run_pipeline import fit_pipeline


def test_fit_fedot_pipeline_classification():
    config_file = os.path.join(fedot_project_root(),
                               'test', 'data', './remote/remote_config_class')
    status = fit_pipeline(config_file)
    assert status is True


def test_fit_fedot_pipeline_time_series():
    config_file = os.path.join(fedot_project_root(),
                               'test', 'data', 'remote', 'remote_config_ts')
    status = fit_pipeline(config_file)
    assert status is True


def test_fit_fedot_pipeline_multivar_time_series():
    config_file = os.path.join(fedot_project_root(),
                               'test', 'data', 'remote', 'remote_config_ts_multivar')
    status = fit_pipeline(config_file)
    assert status is True


def test_pseudo_remote_composer_classification():
    connect_params = {}
    exec_params = {
        'container_input_path': "./test/data/",
        'container_output_path': "./test/data/remote/",
        'container_config_path': "./test/data/remote_config_class",
        'container_image': "test",
        'timeout': 1
    }

    remote_task_params = RemoteTaskParams(
        dataset_name='advanced_classification')

    client = TestClient(connect_params, exec_params, output_path='./test/data/remote')

    RemoteEvaluator().clean()
    _ = RemoteEvaluator(
        client=client,
        remote_task_params=remote_task_params
    )

    composer_params = {
        'pop_size': 3,
        'timeout': 0.01,
        'cv_folds': None
    }

    preset = 'light'
    automl = Fedot(problem='classification', preset=preset, composer_params=composer_params)

    path = os.path.join(fedot_project_root(), 'test', 'data', 'advanced_classification.csv')

    automl.fit(path)
    predict = automl.predict(path)
    assert predict is not None


def test_pseudo_remote_composer_ts_forecasting():
    connect_params = {}
    exec_params = {
        'container_input_path': "./test/data/",
        'container_output_path': "./test/data/remote/",
        'container_config_path': "./test/data/remote_config_ts",
        'container_image': "test",
        'timeout': 1
    }

    remote_task_params = RemoteTaskParams(
        dataset_name='short_time_series')

    client = TestClient(connect_params, exec_params, output_path='./test/data/remote')

    RemoteEvaluator().clean()
    _ = RemoteEvaluator(
        client=client,
        remote_task_params=remote_task_params
    )

    composer_params = {
        'pop_size': 10,
        'timeout': 0.01,
        'cv_folds': None
    }

    preset = 'light'
    automl = Fedot(problem='ts_forecasting', preset=preset, composer_params=composer_params,
                   task_params=TsForecastingParams(forecast_length=1))

    path = os.path.join(fedot_project_root(), 'test', 'data', 'short_time_series.csv')

    automl.fit(path, target='sea_height')
    predict = automl.predict(path)
    assert predict is not None
