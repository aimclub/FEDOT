import os
import random
from datetime import timedelta

import numpy as np

from cases.credit_scoring_problem import run_credit_scoring_problem
from cases.metocean_forecasting_problem import run_metocean_forecasting_problem
from fedot.core.utils import project_root
from sklearn.metrics import mean_squared_error


random.seed(1)
np.random.seed(1)


def test_credit_scoring_problem():
    test_file_path = str(os.path.dirname(__file__))
    file_path_train = os.path.join(test_file_path, '../data/simple_classification.csv')
    file_path_test = file_path_train
    full_path_train = os.path.join(str(project_root()), file_path_train)
    full_path_test = os.path.join(str(project_root()), file_path_test)

    roc_auc_test = run_credit_scoring_problem(full_path_train, full_path_test,
                                              max_lead_time=timedelta(minutes=0.1))
    assert roc_auc_test > 0.5


def test_metocean_forecasting_problem():
    test_file_path = str(os.path.dirname(__file__))
    file_path_train = os.path.join(test_file_path, '../data/simple_time_series.csv')
    file_path_test = file_path_train
    full_path_train = os.path.join(str(project_root()), file_path_train)
    full_path_test = os.path.join(str(project_root()), file_path_test)

    rmse = run_metocean_forecasting_problem(full_path_train, full_path_test,
                                            forecast_length=1, max_window_size=1)
    assert rmse < 50


def test_gapfiilling_problem():
    # Filling in the gaps in column "with_gap"
    dataframe = run_gapfilling_case()

    gap_array = np.array(dataframe['with_gap'])
    gap_ids = np.argwhere(gap_array == -100.0)

    actual = np.array(dataframe['temperature'])[gap_ids]
    ridge_predicted = np.array(dataframe['ridge'])[gap_ids]
    composite_predicted = np.array(dataframe['composite'])[gap_ids]

    rmse_ridge = mean_squared_error(actual, ridge_predicted, squared=False)
    rmse_composite = mean_squared_error(actual, composite_predicted, squared=False)

    assert rmse_ridge < 5.0
    assert rmse_composite < 6.0
