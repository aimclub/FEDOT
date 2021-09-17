import os

from fedot.core.utils import fedot_project_root
from remote.run_pipeline import fit_pipeline


def test_fit_fedot_pipeline_classification():
    config_file = os.path.join(fedot_project_root(),
                               'test', 'data', './remote/remote_config_class')
    status = fit_pipeline(config_file)
    assert status


def test_fit_fedot_pipeline_time_series():
    config_file = os.path.join(fedot_project_root(),
                               'test', 'data', './remote/remote_config_ts')
    status = fit_pipeline(config_file)
    assert status


def test_fit_fedot_pipeline_multivar_time_series():
    config_file = os.path.join(fedot_project_root(),
                               'test', 'data', './remote/remote_config_ts_multivar')
    status = fit_pipeline(config_file)
    assert status
