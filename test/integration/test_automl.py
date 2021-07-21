import os
from datetime import timedelta

import pytest

from examples.pipeline_from_automl import run_pipeline_from_automl
from examples.tpot_vs_fedot import run_tpot_vs_fedot_example
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.utils import fedot_project_root


@pytest.mark.skip('AutoMl models are not supported for now')
def test_pipeline_from_automl_example():
    project_root_path = str(fedot_project_root())
    with OperationTypesRepository().assign_repo('model', 'model_repository_with_automl.json') as _:
        file_path_train = os.path.join(project_root_path, 'test/data/simple_classification.csv')
        file_path_test = file_path_train

        auc = run_pipeline_from_automl(file_path_train, file_path_test, max_run_time=timedelta(seconds=1))
    OperationTypesRepository.assign_repo('model', 'model_repository.json')

    assert auc > 0.5


def test_tpot_vs_fedot_example():
    project_root_path = str(fedot_project_root())
    file_path_train = os.path.join(project_root_path, 'test/data/simple_classification.csv')
    file_path_test = file_path_train

    auc = run_tpot_vs_fedot_example(file_path_train, file_path_test)
    assert auc > 0.5
