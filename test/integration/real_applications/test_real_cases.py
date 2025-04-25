import numpy as np
from golem.core.tuning.simultaneous import SimultaneousTuner
from sklearn.metrics import mean_squared_error

from examples.real_cases.credit_scoring.credit_scoring_problem import run_credit_scoring_problem
from examples.real_cases.metocean_forecasting_problem import run_metocean_forecasting_problem
from examples.real_cases.river_levels_prediction.river_level_case_manual import run_river_experiment
from examples.real_cases.spam_detection import run_text_problem_from_saved_meta_file
from examples.real_cases.time_series_gapfilling_case import run_gapfilling_case
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utils import fedot_project_root


def test_credit_scoring_problem():
    full_path_train = full_path_test = fedot_project_root().joinpath('test/data/simple_classification.csv')

    roc_auc_test = run_credit_scoring_problem(full_path_train, full_path_test, timeout=5, target='Y', n_jobs=1)
    assert roc_auc_test > 0.5


def test_metocean_forecasting_problem():
    full_path_train = fedot_project_root().joinpath('test/data/simple_time_series.csv')
    full_path_test = full_path_train

    rmse = run_metocean_forecasting_problem(full_path_train,
                                            full_path_test,
                                            forecast_length=2,
                                            timeout=0.1)
    print(rmse)
    assert rmse['rmse'] < 500


def test_gapfilling_problem():
    # Filling in the gaps in column "with_gap"
    dataframe = run_gapfilling_case('test/data/simple_ts_gapfilling.csv')

    gap_array = np.array(dataframe['with_gap'])
    gap_ids = np.argwhere(gap_array == -100.0)

    actual = np.array(dataframe['temperature'])[gap_ids]
    ridge_predicted = np.array(dataframe['ridge'])[gap_ids]
    composite_predicted = np.array(dataframe['composite'])[gap_ids]

    rmse_ridge = mean_squared_error(actual, ridge_predicted) ** 0.5
    rmse_composite = mean_squared_error(actual, composite_predicted) ** 0.5

    assert rmse_ridge < 40.0
    assert rmse_composite < 40.0


def test_river_levels_problem():
    # Initialise pipeline for river levels prediction
    node_encoder = PipelineNode('one_hot_encoding')
    node_scaling = PipelineNode('scaling', nodes_from=[node_encoder])
    node_ridge = PipelineNode('ridge', nodes_from=[node_scaling])
    node_lasso = PipelineNode('lasso', nodes_from=[node_scaling])
    node_final = PipelineNode('rfr', nodes_from=[node_ridge, node_lasso])

    init_pipeline = Pipeline(node_final)

    file_path_train = fedot_project_root().joinpath('test/data/station_levels.csv')

    run_river_experiment(file_path=file_path_train,
                         pipeline=init_pipeline,
                         iterations=1,
                         tuner=SimultaneousTuner,
                         tuner_iterations=10)

    is_experiment_finished = True

    assert is_experiment_finished


def test_spam_detection_problem():
    """ Simple launch of spam detection case """
    file_path_train = fedot_project_root().joinpath('test/data/spam_detection.csv')

    # Classification task based on text data
    run_text_problem_from_saved_meta_file(file_path_train)
