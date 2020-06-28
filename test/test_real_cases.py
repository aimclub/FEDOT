import os
import random
from datetime import timedelta

import numpy as np
import pytest

from benchmark.benchmark_model_types import BenchmarkModelTypesEnum
from benchmark.benchmark_utils import get_cancer_case_data_paths
from benchmark.executor import CaseExecutor, ExecutionParams
from cases.credit_scoring_problem import run_credit_scoring_problem
from cases.metocean_forecasting_problem import run_metocean_forecasting_problem
from core.repository.tasks import TaskTypesEnum
from core.utils import project_root

random.seed(1)
np.random.seed(1)


def test_credit_scoring_problem():
    test_file_path = str(os.path.dirname(__file__))
    file_path_train = os.path.join(test_file_path, 'data/simple_classification.csv')
    file_path_test = file_path_train
    full_path_train = os.path.join(str(project_root()), file_path_train)
    full_path_test = os.path.join(str(project_root()), file_path_test)

    roc_auc_test = run_credit_scoring_problem(full_path_train, full_path_test,
                                              max_lead_time=timedelta(minutes=0.1))
    assert roc_auc_test > 0.5


def test_metocean_forecasting_scoring_problem():
    test_file_path = str(os.path.dirname(__file__))
    file_path_train = os.path.join(test_file_path, 'data/simple_time_series.csv')
    file_path_test = file_path_train
    full_path_train = os.path.join(str(project_root()), file_path_train)
    full_path_test = os.path.join(str(project_root()), file_path_test)

    rmse = run_metocean_forecasting_problem(full_path_train, full_path_test,
                                            forecast_length=1, max_window_size=1)
    assert rmse < 50


@pytest.mark.skip('Max run time should be implemented for the caseExecutor')
def test_benchmarks_executor():
    train_file, test_file = get_cancer_case_data_paths()

    result_metrics = CaseExecutor(params=ExecutionParams(train_file=train_file,
                                                         test_file=test_file,
                                                         task=TaskTypesEnum.classification,
                                                         case_label='cancer',
                                                         target_name='target'),
                                  models=[BenchmarkModelTypesEnum.fedot,
                                          BenchmarkModelTypesEnum.baseline],
                                  metric_list=['roc_auc', 'f1']).execute()

    assert result_metrics is not None
