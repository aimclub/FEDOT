import os
from datetime import timedelta

from examples.forecasting_model_composing import run_metocean_forecasting_problem
from examples.multiclass_prediction import get_model
from examples.time_series_forecasting import (run_multistep_composite_example, run_multistep_linear_example,
                                              run_multistep_lstm_example, run_multistep_multiscale_example,
                                              run_onestep_linear_example)


def test_forecasting_model_composing_example():
    test_file_path = str(os.path.dirname(__file__))
    file_path_train = os.path.join(test_file_path, '../data/simple_time_series.csv')
    file_path_test = file_path_train

    rmse = run_metocean_forecasting_problem(file_path_train, file_path_test, with_visualisation=False)
    assert rmse > 0


def test_ts_forecasting_example():
    data_length = 700
    data_length_onestep = 64
    run_onestep_linear_example(n_steps=data_length_onestep, is_visualise=False)
    run_multistep_linear_example(n_steps=data_length, is_visualise=False)
    run_multistep_multiscale_example(n_steps=data_length, is_visualise=False)
    run_multistep_composite_example(n_steps=data_length, is_visualise=False)
    run_multistep_lstm_example(n_steps=data_length, is_visualise=False)


def test_multiclass_example():
    test_file_path = str(os.path.dirname(__file__))
    file_path_train = os.path.join(test_file_path, '../data/multiclass_classification.csv')

    chain = get_model(file_path_train, cur_lead_time=timedelta(seconds=1))
    assert chain is not None
