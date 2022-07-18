import os
from datetime import timedelta
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error

from examples.advanced.multi_modal_pipeline import run_multi_modal_pipeline
from examples.advanced.multiobj_optimisation import run_classification_multiobj_example
from examples.advanced.time_series_forecasting.multistep import run_multistep
from examples.advanced.time_series_forecasting.nemo_multiple import run_multiple_example
from examples.simple.classification.api_classification import run_classification_example
from examples.simple.classification.classification_pipelines import classification_complex_pipeline
from examples.simple.classification.multiclass_prediction import get_model
from examples.simple.interpretable.api_explain import run_api_explain_example
from examples.simple.pipeline_log import run_log_example
from examples.simple.pipeline_tune import get_case_train_test_data, pipeline_tuning
from examples.advanced.time_series_forecasting.exogenous import run_exogenous_experiment
from examples.simple.time_series_forecasting.api_forecasting import run_ts_forecasting_example
from examples.simple.time_series_forecasting.gapfilling import run_gapfilling_example
from examples.simple.time_series_forecasting.ts_pipelines import ts_complex_dtreg_pipeline

from fedot.core.utils import fedot_project_root, default_fedot_data_dir


def test_multiclass_example():
    project_root_path = str(fedot_project_root())
    file_path_train = os.path.join(project_root_path, 'test/data/multiclass_classification.csv')

    pipeline = get_model(file_path_train, cur_lead_time=timedelta(seconds=5))
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


def test_log_example():
    with open(Path(default_fedot_data_dir(), 'log.log')) as f:
        lines_before = len(f.readlines())
    run_log_example('log.log')
    with open(Path(default_fedot_data_dir(), 'log.log')) as f:
        lines_after = len(f.readlines())
    assert lines_after > lines_before


def test_pipeline_tuning_example():
    train_data, test_data = get_case_train_test_data()

    # Pipeline composition
    pipeline = classification_complex_pipeline()

    # Pipeline tuning
    after_tune_roc_auc, _ = pipeline_tuning(pipeline=pipeline,
                                            train_data=train_data,
                                            test_data=test_data,
                                            local_iter=1,
                                            tuner_iter_num=2)


def test_multistep_example():
    pipeline = ts_complex_dtreg_pipeline()
    run_multistep('test_sea', pipeline, step_forecast=20, future_steps=5)


def test_api_example():
    prediction = run_classification_example(timeout=1)
    assert prediction is not None

    forecast = run_ts_forecasting_example(dataset='australia', timeout=1)
    assert forecast is not None

    pareto = run_classification_multiobj_example(timeout=1)
    assert pareto is not None

    explainer = run_api_explain_example(visualize=False, timeout=1)
    assert explainer is not None


def test_multi_modal_example():
    result = run_multi_modal_pipeline(files_path='cases/data/mm_imdb', is_visualise=False)
    assert result > 0
