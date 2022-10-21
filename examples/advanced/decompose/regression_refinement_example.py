import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root

warnings.filterwarnings('ignore')


def get_refinement_pipeline():
    """ Create five-level pipeline with decompose operation """
    node_encoding = PrimaryNode('one_hot_encoding')
    node_scaling = SecondaryNode('scaling', nodes_from=[node_encoding])
    node_lasso = SecondaryNode('lasso', nodes_from=[node_scaling])
    node_decompose = SecondaryNode('decompose', nodes_from=[node_scaling, node_lasso])
    node_dtreg = SecondaryNode('dtreg', nodes_from=[node_decompose])
    node_dtreg.parameters = {'max_depth': 3}
    final_node = SecondaryNode('ridge', nodes_from=[node_lasso, node_dtreg])

    pipeline = Pipeline(final_node)
    return pipeline


def get_non_refinement_pipeline():
    node_encoding = PrimaryNode('one_hot_encoding')
    node_scaling = SecondaryNode('scaling', nodes_from=[node_encoding])

    node_lasso = SecondaryNode('lasso', nodes_from=[node_scaling])
    node_dtreg = SecondaryNode('dtreg', nodes_from=[node_scaling])
    node_dtreg.parameters = {'max_depth': 3}

    final_node = SecondaryNode('ridge', nodes_from=[node_lasso, node_dtreg])

    pipeline = Pipeline(final_node)
    return pipeline


def prepare_input_data(features, target):
    """ Function create InputData with features """
    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
        features,
        target,
        test_size=0.2,
        shuffle=True,
        random_state=10)
    y_data_test = np.ravel(y_data_test)

    # Define regression task
    task = Task(TaskTypesEnum.regression)

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(x_data_train)),
                            features=x_data_train,
                            target=y_data_train,
                            task=task,
                            data_type=DataTypesEnum.table)

    predict_input = InputData(idx=np.arange(0, len(x_data_test)),
                              features=x_data_test,
                              target=y_data_test,
                              task=task,
                              data_type=DataTypesEnum.table)

    return train_input, predict_input, task


def run_river_experiment(file_path, with_tuning=False):
    """ Function launch example with experimental features of the FEDOT framework

    :param file_path: path to the csv file
    :param with_tuning: is it need to tune pipelines or not
    """

    # Read dataframe and prepare train and test data
    df = pd.read_csv(file_path)
    features = np.array(df[['level_station_1', 'mean_temp', 'month', 'precip']])
    target = np.array(df['level_station_2']).reshape((-1, 1))

    # Prepare InputData for train and test
    train_input, predict_input, task = prepare_input_data(features, target)
    y_data_test = predict_input.target

    # Get refinement pipeline
    r_pipeline = get_refinement_pipeline()
    non_pipeline = get_non_refinement_pipeline()

    if with_tuning:
        tuner = TunerBuilder(task)\
            .with_tuner(PipelineTuner)\
            .with_metric(RegressionMetricsEnum.MAE)\
            .with_iterations(100)\
            .build(train_input)
        r_pipeline = tuner.tune(r_pipeline)
        non_pipeline = tuner.tune(non_pipeline)

    # Fit it
    r_pipeline.fit(train_input)
    non_pipeline.fit(train_input)

    # Predict
    predicted_values = r_pipeline.predict(predict_input)
    r_preds = predicted_values.predict

    # Predict
    predicted_values = non_pipeline.predict(predict_input)
    non_preds = predicted_values.predict

    y_data_test = np.ravel(y_data_test)

    mse_value = mean_squared_error(y_data_test, r_preds, squared=False)
    mae_value = mean_absolute_error(y_data_test, r_preds)
    print(f'RMSE with decompose - {mse_value:.2f}')
    print(f'MAE with decompose - {mae_value:.2f}\n')

    mse_value_non = mean_squared_error(y_data_test, non_preds, squared=False)
    mae_value_non = mean_absolute_error(y_data_test, non_preds)
    print(f'RMSE non decompose - {mse_value_non:.2f}')
    print(f'MAE non decompose - {mae_value_non:.2f}\n')


if __name__ == '__main__':
    run_river_experiment(file_path=f'{fedot_project_root()}/cases/data/river_levels/station_levels.csv',
                         with_tuning=True)
