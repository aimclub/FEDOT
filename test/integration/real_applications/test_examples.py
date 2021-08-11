import os
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from examples.fedot_api_example import (run_classification_example, run_classification_multiobj_example,
                                        run_ts_forecasting_example)
from examples.multi_modal_pipeline import run_multi_modal_pipeline
from examples.multiclass_prediction import get_model
from examples.pipeline_and_history_visualisation import run_pipeline_ang_history_visualisation
from examples.pipeline_log import run_log_example
from examples.pipeline_tune import get_case_train_test_data, get_simple_pipeline, pipeline_tuning
from examples.time_series.ts_forecasting_with_exogenous import run_exogenous_experiment
from examples.time_series.ts_foresting_with_nemo_multiple_example import run_multiple_example
from examples.time_series.ts_gapfilling_example import run_gapfilling_example
from examples.time_series.ts_multistep_example import run_multistep_example
from fedot.core.utils import fedot_project_root


def test_multiclass_example():
    project_root_path = str(fedot_project_root())
    file_path_train = os.path.join(project_root_path, 'test/data/multiclass_classification.csv')

    pipeline = get_model(file_path_train, cur_lead_time=timedelta(seconds=1))
    assert pipeline is not None


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
    project_root_path = str(fedot_project_root())
    path = os.path.join(project_root_path, 'test/data/simple_sea_level.csv')
    run_exogenous_experiment(path_to_file=path,
                             len_forecast=50, with_exog=True,
                             with_visualisation=False)


def test_nemo_multiple_points_example():
    project_root_path = str(fedot_project_root())
    path = os.path.join(project_root_path, 'test/data/ssh_points_grid_simple.csv')
    exog_path = os.path.join(project_root_path, 'test/data/ssh_nemo_points_grid_simple.csv')
    run_multiple_example(path_to_file=path,
                         path_to_exog_file=exog_path,
                         out_path=None,
                         len_forecast=30,
                         is_boxplot_visualize=False)


def test_pipeline_and_history_example():
    run_pipeline_ang_history_visualisation(with_pipeline_visualisation=False)


def test_log_example():
    log_file_name = 'example_log.log'
    run_log_example(log_file_name)

    assert os.path.isfile(log_file_name)


def test_pipeline_tuning_example():
    train_data, test_data = get_case_train_test_data()

    # Pipeline composition
    pipeline = get_simple_pipeline()

    # Pipeline tuning
    after_tune_roc_auc, _ = pipeline_tuning(pipeline=pipeline,
                                            train_data=train_data,
                                            test_data=test_data,
                                            local_iter=1,
                                            tuner_iter_num=2)


def test_multistep_example():
    project_root_path = str(fedot_project_root())
    path = os.path.join(project_root_path, 'test/data/simple_sea_level.csv')

    df = pd.read_csv(path)
    time_series = np.array(df['Level'])

    run_multistep_example(time_series,
                          len_forecast=20,
                          future_steps=40,
                          vis=False)


def test_api_example():
    prediction = run_classification_example(timeout=1)
    assert prediction is not None

    forecast = run_ts_forecasting_example(with_plot=False, with_pipeline_vis=False, timeout=1)
    assert forecast is not None

    pareto = run_classification_multiobj_example(timeout=1)
    assert pareto is not None


def test_multi_modal_example():
    result = run_multi_modal_pipeline(files_path='cases/data/mm_imdb', is_visualise=False)
    assert result > 0
