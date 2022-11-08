import os
import random

import numpy as np
from sklearn.metrics import mean_squared_error

from cases.credit_scoring.credit_scoring_problem import run_credit_scoring_problem
from cases.metocean_forecasting_problem import run_metocean_forecasting_problem
from cases.river_levels_prediction.river_level_case_manual import run_river_experiment
from cases.spam_detection import run_text_problem_from_saved_meta_file
from cases.time_series_gapfilling_case import run_gapfilling_case
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.utils import fedot_project_root

random.seed(1)
np.random.seed(1)


def test_credit_scoring_problem():
    project_root_path = str(fedot_project_root())
    file_path_train = os.path.join(project_root_path, 'test/data/simple_classification.csv')
    file_path_test = file_path_train
    full_path_train = os.path.join(str(fedot_project_root()), file_path_train)
    full_path_test = os.path.join(str(fedot_project_root()), file_path_test)

    roc_auc_test = run_credit_scoring_problem(full_path_train, full_path_test, timeout=5, target='Y')
    assert roc_auc_test > 0.5


def test_metocean_forecasting_problem():
    project_root_path = str(fedot_project_root())
    file_path_train = os.path.join(project_root_path, 'test/data/simple_time_series.csv')
    file_path_test = file_path_train
    full_path_train = os.path.join(str(fedot_project_root()), file_path_train)
    full_path_test = os.path.join(str(fedot_project_root()), file_path_test)

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

    rmse_ridge = mean_squared_error(actual, ridge_predicted, squared=False)
    rmse_composite = mean_squared_error(actual, composite_predicted, squared=False)

    assert rmse_ridge < 40.0
    assert rmse_composite < 40.0


def test_river_levels_problem():
    # Initialise pipeline for river levels prediction
    node_encoder = PrimaryNode('one_hot_encoding')
    node_scaling = SecondaryNode('scaling', nodes_from=[node_encoder])
    node_ridge = SecondaryNode('ridge', nodes_from=[node_scaling])
    node_lasso = SecondaryNode('lasso', nodes_from=[node_scaling])
    node_final = SecondaryNode('rfr', nodes_from=[node_ridge, node_lasso])

    init_pipeline = Pipeline(node_final)

    project_root_path = str(fedot_project_root())
    file_path_train = os.path.join(project_root_path, 'test/data/station_levels.csv')

    run_river_experiment(file_path=file_path_train,
                         pipeline=init_pipeline,
                         iterations=1,
                         tuner=PipelineTuner,
                         tuner_iterations=10)

    is_experiment_finished = True

    assert is_experiment_finished


def test_spam_detection_problem():
    """ Simple launch of spam detection case """
    project_root_path = str(fedot_project_root())
    file_path_train = os.path.join(project_root_path, 'test/data/spam_detection.csv')

    # Classification task based on text data
    run_text_problem_from_saved_meta_file(file_path_train)

