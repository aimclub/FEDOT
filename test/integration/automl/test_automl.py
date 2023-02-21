import os
from datetime import timedelta

from examples.advanced.automl.h2o_example import h2o_classification_pipeline_evaluation, \
    h2o_regression_pipeline_evaluation, h2o_ts_pipeline_evaluation
from examples.advanced.automl.pipeline_from_automl import run_pipeline_from_automl
from examples.advanced.automl.tpot_vs_fedot import run_tpot_vs_fedot_example
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.utils import fedot_project_root


def test_pipeline_from_automl_example():
    project_root_path = str(fedot_project_root())

    file_path_train = os.path.join(project_root_path, 'test/data/simple_classification.csv')
    file_path_test = file_path_train

    auc = run_pipeline_from_automl(file_path_train, file_path_test, max_run_time=timedelta(seconds=1))

    assert auc > 0.5


def test_tpot_vs_fedot_example():
    project_root_path = str(fedot_project_root())
    file_path_train = os.path.join(project_root_path, 'test/data/simple_classification.csv')
    file_path_test = file_path_train

    auc = run_tpot_vs_fedot_example(file_path_train, file_path_test)
    assert auc > 0.5


def test_h2o_vs_fedot_example():
    with OperationTypesRepository.init_automl_repository() as _:
        h2o_classification_pipeline_evaluation()
    with OperationTypesRepository.init_automl_repository() as _:
        h2o_regression_pipeline_evaluation()
    with OperationTypesRepository.init_automl_repository() as _:
        h2o_ts_pipeline_evaluation()

