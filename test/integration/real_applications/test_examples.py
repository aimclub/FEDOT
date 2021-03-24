import os
from datetime import timedelta

import numpy as np
from sklearn.metrics import mean_squared_error

from examples.chain_and_history_visualisation import run_chain_ang_history_visualisation
from examples.chain_log import run_log_example
from examples.chain_tune import chain_tuning, get_case_train_test_data, get_simple_chain
from examples.fedot_api_example import (run_classification_example, run_classification_multiobj_example,
                                        run_ts_forecasting_example)
from examples.multiclass_prediction import get_model
from examples.ts_forecasting_with_exogenous import run_exogenous_experiment
from examples.ts_gapfilling_example import run_gapfilling_example
from fedot.core.utils import project_root


def test_multiclass_example():
    project_root_path = str(project_root())
    file_path_train = os.path.join(project_root_path, 'test/data/multiclass_classification.csv')

    chain = get_model(file_path_train, cur_lead_time=timedelta(seconds=1))
    assert chain is not None


def test_gapfilling_example():
    arrays_dict, gap_data, real_data = run_gapfilling_example()

    gap_ids = np.ravel(np.argwhere(gap_data == -100.0))
    for key in arrays_dict.keys():
        arr_without_gaps = arrays_dict.get(key)

        # Get only values in the gap
        predicted_values = arr_without_gaps[gap_ids]
        true_values = real_data[gap_ids]

        model_rmse = mean_squared_error(true_values, predicted_values, squared=False)
        assert model_rmse < 0.5


def test_exogenous_ts_example():
    project_root_path = str(project_root())
    path = os.path.join(project_root_path, 'test/data/simple_sea_level.csv')
    run_exogenous_experiment(path_to_file=path,
                             len_forecast=50, with_exog=True,
                             with_visualisation=False)


def test_chain_and_history_example():
    run_chain_ang_history_visualisation(with_chain_visualisation=False)


def test_log_example():
    log_file_name = 'example_log.log'
    run_log_example(log_file_name)

    assert os.path.isfile(log_file_name)


def test_chain_tuning_example():
    train_data, test_data = get_case_train_test_data()

    # Chain composition
    chain = get_simple_chain()

    # Chain tuning
    after_tune_roc_auc, _ = chain_tuning(chain=chain,
                                         train_data=train_data,
                                         test_data=test_data,
                                         local_iter=1,
                                         tuner_iter_num=2)


def test_api_example():
    prediction = run_classification_example()
    assert prediction is not None

    forecast = run_ts_forecasting_example(with_plot=False, with_chain_vis=False)
    assert forecast is not None

    pareto = run_classification_multiobj_example()
    assert pareto is not None
