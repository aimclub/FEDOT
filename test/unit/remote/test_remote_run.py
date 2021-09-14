import os

from fedot.core.utils import fedot_project_root
from remote.run_pipeline import fit_pipeline


def test_fit_fedot_pipeline():
    config_file = os.path.join(fedot_project_root(), 'test', 'data', 'remote_config')
    status = fit_pipeline(config_file)
    assert status
