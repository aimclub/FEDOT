import os
import pytest
from datetime import timedelta

from fedot.core.repository.operation_types_repository import OperationTypesRepository
from examples.chain_from_automl import run_chain_from_automl
from examples.tpot_vs_fedot import run_tpot_vs_fedot_example
from fedot.core.utils import project_root


@pytest.mark.skip('AutoMl models are not supported for now')
def test_chain_from_automl_example():
    project_root_path = str(project_root())
    experimental_repo_file = os.path.join('model_repository_with_automl.json')
    with OperationTypesRepository(experimental_repo_file) as _:
        file_path_train = os.path.join(project_root_path, 'test/data/simple_classification.csv')
        file_path_test = file_path_train

        auc = run_chain_from_automl(file_path_train, file_path_test, max_run_time=timedelta(seconds=1))

    assert auc > 0.5


def test_tpot_vs_fedot_example():
    project_root_path = str(project_root())
    file_path_train = os.path.join(project_root_path, 'test/data/simple_classification.csv')
    file_path_test = file_path_train

    auc = run_tpot_vs_fedot_example(file_path_train, file_path_test)
    assert auc > 0.5
