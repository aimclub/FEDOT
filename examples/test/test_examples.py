import os
from datetime import timedelta

from examples.chain_from_automl import run_chain_from_automl
from examples.forecasting_model_composing import run_metocean_forecasting_problem
from examples.multiclass_prediction import get_model
from examples.tpot_vs_fedot import run_tpot_vs_fedot_example
import pytest


@pytest.mark.skip('TPOT excluded for req.txt')
def test_chain_from_automl_example():
    test_file_path = str(os.path.dirname(__file__))
    file_path_train = os.path.join(test_file_path, 'data/simple_classification.csv')
    file_path_test = file_path_train

    auc = run_chain_from_automl(file_path_train, file_path_test, max_run_time=timedelta(seconds=1))
    assert auc > 0.5


def test_tpot_vs_fedot_example():
    test_file_path = str(os.path.dirname(__file__))
    file_path_train = os.path.join(test_file_path, 'data/simple_classification.csv')
    file_path_test = file_path_train

    auc = run_tpot_vs_fedot_example(file_path_train, file_path_test)
    assert auc > 0.5


def test_forecasting_model_composing_example():
    test_file_path = str(os.path.dirname(__file__))
    file_path_train = os.path.join(test_file_path, 'data/simple_time_series.csv')
    file_path_test = file_path_train

    rmse = run_metocean_forecasting_problem(file_path_train, file_path_test, with_visualisation=False)
    assert rmse > 0


def test_multiclass_example():
    test_file_path = str(os.path.dirname(__file__))
    file_path_train = os.path.join(test_file_path, 'data/multiclass_classification.csv')

    chain = get_model(file_path_train, cur_lead_time=timedelta(seconds=1))
    assert chain is not None
