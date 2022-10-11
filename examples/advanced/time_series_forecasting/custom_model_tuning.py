import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hyperopt import hp
from sklearn.linear_model import Ridge

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.search_space import SearchSpace
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task, TsForecastingParams


# implementation of custom model without fitting
def domain_model_imitation_predict(fitted_model: any, idx: np.array, predict_data: np.array, params: dict):
    # TODO real custom model or more realistic imitation
    a = params.get('a')
    b = params.get('b')
    shape = predict_data.shape
    result = np.random.rand(*shape) * a + b
    # Available output_type's 'table', 'ts', 'image', 'text'
    return result, 'table'


# implementation of custom regression model imitation (fit)
def custom_ml_model_imitation_fit(idx: np.array, features: np.array, target: np.array, params: dict):
    alpha = params.get('alpha')
    reg = Ridge(alpha=alpha)
    reg.fit(features, target)
    return reg


# implementation of custom regression model imitation (predict)
def custom_ml_model_imitation_predict(fitted_model: any, idx: np.array, features: np.array, params: dict):
    res = fitted_model.predict(features)
    return res, 'table'


def get_domain_pipeline():
    """
        Pipeline looking like this
        lagged -> custom -> ridge
    """
    lagged_node = PrimaryNode('lagged')
    lagged_node.parameters = {'window_size': 50}

    # For custom model params as initial approximation and model as function is necessary
    custom_node = SecondaryNode('custom', nodes_from=[lagged_node])
    custom_node.parameters = {"a": -50, "b": 500, 'model_predict': domain_model_imitation_predict}

    node_final = SecondaryNode('ridge', nodes_from=[custom_node])
    pipeline = Pipeline(node_final)

    return pipeline


def get_fitting_custom_pipeline():
    """
        Pipeline looking like this
        lagged -> custom -> ridge
    """
    lagged_node = PrimaryNode('lagged')
    lagged_node.parameters = {'window_size': 50}

    # For custom model params as initial approximation and model as function is necessary
    custom_node = SecondaryNode('custom', nodes_from=[lagged_node])
    custom_node.parameters = {'alpha': 5,
                                 'model_predict': custom_ml_model_imitation_predict,
                                 'model_fit': custom_ml_model_imitation_fit}

    node_final = SecondaryNode('lasso', nodes_from=[custom_node])
    pipeline = Pipeline(node_final)

    return pipeline


def run_pipeline_tuning(time_series, len_forecast, pipeline_type):
    # Source time series
    train_input, predict_input = train_test_data_setup(InputData(idx=range(len(time_series)),
                                                                 features=time_series,
                                                                 target=time_series,
                                                                 task=Task(TaskTypesEnum.ts_forecasting,
                                                                           TsForecastingParams(
                                                                               forecast_length=len_forecast)),
                                                                 data_type=DataTypesEnum.ts))

    if pipeline_type == 'with_fit':
        pipeline = get_fitting_custom_pipeline()
        # Setting custom search space for tuner (necessary)
        # model and output_type should be wrapped into hyperopt
        custom_search_space = {'custom': {
            'alpha': (hp.uniform, [0.01, 10]),
            'model_predict': (hp.choice, [[custom_ml_model_imitation_predict]]),
            'model_fit': (hp.choice, [[custom_ml_model_imitation_fit]])}}
    elif pipeline_type == 'without_fit':
        pipeline = get_domain_pipeline()
        # Setting custom search space for tuner (necessary)
        # model and output_type should be wrapped into hyperopt
        custom_search_space = {'custom': {'a': (hp.uniform, [-100, 100]),
                                          'b': (hp.uniform, [0, 1000]),
                                          'model_predict': (hp.choice, [[domain_model_imitation_predict]])}}
    pipeline.fit_from_scratch(train_input)
    pipeline.print_structure()
    # Get prediction with initial approximation
    predicted_before_tuning = pipeline.predict(predict_input).predict

    replace_default_search_space = True
    cv_folds = 3
    validation_blocks = 3
    search_space = SearchSpace(custom_search_space=custom_search_space,
                               replace_default_search_space=replace_default_search_space)
    pipeline_tuner = TunerBuilder(train_input.task)\
        .with_tuner(PipelineTuner)\
        .with_metric(RegressionMetricsEnum.RMSE)\
        .with_cv_folds(cv_folds)\
        .with_validation_blocks(validation_blocks)\
        .with_iterations(10)\
        .with_search_space(search_space).build(train_input)
    # Tuning pipeline
    pipeline = pipeline_tuner.tune(pipeline)
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
    df = pd.read_csv('../../../cases/data/time_series/metocean.csv')
    time_series = np.array(df['value'])
    run_pipeline_tuning(time_series=time_series,
                        len_forecast=50,
                        pipeline_type='with_fit')  # mean custom ml model with fit, 'without_fit' - means domain model
