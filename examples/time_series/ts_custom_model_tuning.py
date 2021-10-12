import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hyperopt import hp
from sklearn.metrics import mean_squared_error
from fedot.core.pipelines.tuning.search_space import SearchSpace

from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.unified import PipelineTuner
from examples.time_series.ts_forecasting_tuning import prepare_input_data


def custom_model_imitation(train_data, test_data, params):
    # TODO real custom model or more realistic imitation
    a = params.get('a')
    b = params.get('b')
    shape = train_data.shape
    out_type = 'ts'
    if len(shape) > 1:
        out_type = 'table'
    result = np.random.rand(*shape) * a + b
    return result, out_type


def get_simple_pipeline():
    """
        Pipeline looking like this
        lagged -> custom -> ridge
    """
    lagged_node = PrimaryNode('lagged')
    lagged_node.custom_params = {'window_size': 50}

    # For custom model params as initial approximation and wrappers with custom model as function is necessary
    custom_node = SecondaryNode('custom', nodes_from=[lagged_node])
    custom_node.custom_params = {"a": -50, "b": 500, 'model': custom_model_imitation}

    node_final = SecondaryNode('ridge', nodes_from=[custom_node])
    pipeline = Pipeline(node_final)

    return pipeline


def run_pipeline_tuning(time_series, len_forecast):
    len_forecast = len_forecast
    train_data = time_series[:-len_forecast]
    test_data = time_series[-len_forecast:]

    # Source time series
    train_input, predict_input, task = prepare_input_data(len_forecast=len_forecast,
                                                          train_data_features=train_data,
                                                          train_data_target=train_data,
                                                          test_data_features=train_data)
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
                                   task=task,
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
    plt.plot(np.arange(len(test_data)), test_data, label='Real')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('../../cases/data/time_series/metocean.csv')
    time_series = np.array(df['value'])
    run_pipeline_tuning(time_series=time_series,
                        len_forecast=50)
