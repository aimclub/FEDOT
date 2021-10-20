import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hyperopt import hp
from sklearn.metrics import mean_squared_error

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.search_space import SearchSpace
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task, TsForecastingParams


def custom_model_imitation(train_data, test_data, params):
    # TODO real custom model or more realistic imitation
    a = params.get('a')
    b = params.get('b')
    shape = train_data.shape
    result = np.random.rand(*shape) * a + b
    # Available output_type's 'table', 'ts', 'image', 'text'
    return result, 'table'


def get_simple_pipeline():
    """
        Pipeline looking like this
        lagged -> custom -> ridge
    """
    lagged_node = PrimaryNode('lagged')
    lagged_node.custom_params = {'window_size': 50}

    # For custom model params as initial approximation and model as function is necessary
    custom_node = SecondaryNode('custom', nodes_from=[lagged_node])
    custom_node.custom_params = {"a": -50, "b": 500, 'model': custom_model_imitation}

    node_final = SecondaryNode('ridge', nodes_from=[custom_node])
    pipeline = Pipeline(node_final)

    return pipeline


def run_pipeline_tuning(time_series, len_forecast):
    # Source time series
    train_input, predict_input = train_test_data_setup(InputData(idx=range(len(time_series)),
                                                                 features=time_series,
                                                                 target=time_series,
                                                                 task=Task(TaskTypesEnum.ts_forecasting,
                                                                           TsForecastingParams(
                                                                               forecast_length=len_forecast)),
                                                                 data_type=DataTypesEnum.ts))
    pipeline = get_simple_pipeline()
    pipeline.fit_from_scratch(train_input)
    pipeline.print_structure()
    # Get prediction with initial approximation
    predicted_before_tuning = pipeline.predict(predict_input).predict

    # Setting custom search space for tuner (necessary)
    # model and output_type should be wrapped into hyperopt
    custom_search_space = {'custom': {'a': (hp.uniform, [-100, 100]),
                                      'b': (hp.uniform, [0, 1000]),
                                      'model': (hp.choice, [[custom_model_imitation]])}}
    replace_default_search_space = True
    pipeline_tuner = PipelineTuner(pipeline=pipeline,
                                   task=train_input.task,
                                   iterations=10,
                                   search_space=SearchSpace(custom_search_space=custom_search_space,
                                                            replace_default_search_space=replace_default_search_space))
    # Tuning pipeline
    pipeline = pipeline_tuner.tune_pipeline(input_data=train_input,
                                            loss_function=mean_squared_error,
                                            loss_params={'squared': False},
                                            cv_folds=3,
                                            validation_blocks=3)
    # Fit pipeline on the entire train data
    pipeline.fit_from_scratch(train_input)
    # Predict tuned pipeline
    predicted_values = pipeline.predict(predict_input).predict
    pipeline.print_structure()

    plt.plot(np.arange(len(predicted_before_tuning[0])), predicted_before_tuning[0], label='Before tuning')
    plt.plot(np.arange(len(predicted_values[0])), predicted_values[0], label='After tuning')
    plt.plot(np.arange(len(predict_input.target)), predict_input.target, label='Real')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('../../cases/data/time_series/metocean.csv')
    time_series = np.array(df['value'])
    run_pipeline_tuning(time_series=time_series,
                        len_forecast=50)
