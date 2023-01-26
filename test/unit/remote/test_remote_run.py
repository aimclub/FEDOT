import os
import shutil
from pathlib import Path

import pytest

from fedot.core.utils import fedot_project_root
from fedot.remote.remote_evaluator import RemoteEvaluator, RemoteTaskParams
from fedot.remote.run_pipeline import fit_pipeline


@pytest.fixture(autouse=True)
def run_around_tests():
    yield
    # return evaluator to local mode
    evaluator = RemoteEvaluator()
    evaluator.init(None, RemoteTaskParams(mode='local'))


def test_fit_fedot_pipeline_classification():
    config_file = os.path.join(fedot_project_root(),
                               'test', 'data', 'remote', 'remote_config_class')
    status = fit_pipeline(config_file, save_pipeline=False)
    assert status is True


def test_fit_fedot_pipeline_time_series():
    config_file = os.path.join(fedot_project_root(),
                               'test', 'data', 'remote', 'remote_config_ts')
    status = fit_pipeline(config_file, save_pipeline=False)
    assert status is True


def test_fit_fedot_pipeline_multivar_time_series():
    config_file = os.path.join(fedot_project_root(),
                               'test', 'data', 'remote', 'remote_config_ts_multivar')
    status = fit_pipeline(config_file, save_pipeline=False)
    assert status is True
