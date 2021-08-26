from random import seed

import numpy as np
from sklearn.metrics import mean_squared_error as mse, roc_auc_score as roc

from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.sequential import SequentialTuner
from fedot.core.pipelines.tuning.unified import PipelineTuner
from data.data_manager import get_ts_data, classification_dataset, regression_dataset
from data.pipeline_manager import get_simple_regr_pipeline, get_complex_regr_pipeline,\
    get_simple_class_pipeline, get_complex_class_pipeline

seed(1)
np.random.seed(1)


def test_custom_params_setter():
    data = classification_dataset()
    pipeline = get_complex_class_pipeline()

    custom_params = dict(C=10)

    pipeline.root_node.custom_params = custom_params
    pipeline.fit(data)
    params = pipeline.root_node.fitted_operation.get_params()

    assert params['C'] == 10


def test_pipeline_tuner_regression_correct():
    """ Test PipelineTuner for pipeline based on hyperopt library """
    data = regression_dataset()
    train_data, test_data = train_test_data_setup(data=data)

    # Pipelines for regression task
    pipeline_simple = get_simple_regr_pipeline()
    pipeline_complex = get_complex_regr_pipeline()

    for pipeline in [pipeline_simple, pipeline_complex]:
        # Pipeline tuning
        pipeline_tuner = PipelineTuner(pipeline=pipeline,
                                       task=train_data.task,
                                       iterations=1)
        # Optimization will be performed on RMSE metric, so loss params are defined
        tuned_pipeline = pipeline_tuner.tune_pipeline(input_data=train_data,
                                                      loss_function=mse,
                                                      loss_params={'squared': False})
    is_tuning_finished = True

    assert is_tuning_finished


def test_pipeline_tuner_classification_correct():
    """ Test PipelineTuner for pipeline based on hyperopt library """
    data = classification_dataset()
    train_data, test_data = train_test_data_setup(data=data)

    # Pipelines for classification task
    pipeline_simple = get_simple_class_pipeline()
    pipeline_complex = get_complex_class_pipeline()

    for pipeline in [pipeline_simple, pipeline_complex]:
        # Pipeline tuning
        pipeline_tuner = PipelineTuner(pipeline=pipeline,
                                       task=train_data.task,
                                       iterations=1)
        tuned_pipeline = pipeline_tuner.tune_pipeline(input_data=train_data,
                                                      loss_function=roc)
    is_tuning_finished = True

    assert is_tuning_finished


def test_sequential_tuner_regression_correct():
    """ Test SequentialTuner for pipeline based on hyperopt library """
    data = regression_dataset()
    train_data, test_data = train_test_data_setup(data=data)

    # Pipelines for regression task
    pipeline_simple = get_simple_regr_pipeline()
    pipeline_complex = get_complex_regr_pipeline()

    for pipeline in [pipeline_simple, pipeline_complex]:
        # Pipeline tuning
        sequential_tuner = SequentialTuner(pipeline=pipeline,
                                           task=train_data.task,
                                           iterations=1)
        # Optimization will be performed on RMSE metric, so loss params are defined
        tuned_pipeline = sequential_tuner.tune_pipeline(input_data=train_data,
                                                        loss_function=mse,
                                                        loss_params={'squared': False})
    is_tuning_finished = True

    assert is_tuning_finished


def test_sequential_tuner_classification_correct():
    """ Test SequentialTuner for pipeline based on hyperopt library """
    data = classification_dataset()
    train_data, test_data = train_test_data_setup(data=data)

    # Pipelines for classification task
    pipeline_simple = get_simple_class_pipeline()
    pipeline_complex = get_complex_class_pipeline()

    for pipeline in [pipeline_simple, pipeline_complex]:
        # Pipeline tuning
        sequential_tuner = SequentialTuner(pipeline=pipeline,
                                           task=train_data.task,
                                           iterations=2)
        tuned_pipeline = sequential_tuner.tune_pipeline(input_data=train_data,
                                                        loss_function=roc)
    is_tuning_finished = True

    assert is_tuning_finished


def test_certain_node_tuning_regression_correct():
    """ Test SequentialTuner for particular node based on hyperopt library """
    data = regression_dataset()
    train_data, test_data = train_test_data_setup(data=data)

    # Pipelines for regression task
    pipeline_simple = get_simple_regr_pipeline()
    pipeline_complex = get_complex_regr_pipeline()

    for pipeline in [pipeline_simple, pipeline_complex]:
        # Pipeline tuning
        sequential_tuner = SequentialTuner(pipeline=pipeline,
                                           task=train_data.task,
                                           iterations=1)
        tuned_pipeline = sequential_tuner.tune_node(input_data=train_data,
                                                    node_index=0,
                                                    loss_function=mse)
    is_tuning_finished = True

    assert is_tuning_finished


def test_certain_node_tuning_classification_correct():
    """ Test SequentialTuner for particular node based on hyperopt library """
    data = classification_dataset()
    train_data, test_data = train_test_data_setup(data=data)

    # Pipelines for classification task
    pipeline_simple = get_simple_class_pipeline()
    pipeline_complex = get_complex_class_pipeline()

    for pipeline in [pipeline_simple, pipeline_complex]:
        # Pipeline tuning
        sequential_tuner = SequentialTuner(pipeline=pipeline,
                                           task=train_data.task,
                                           iterations=1)
        tuned_pipeline = sequential_tuner.tune_node(input_data=train_data,
                                                    node_index=0,
                                                    loss_function=roc)
    is_tuning_finished = True

    assert is_tuning_finished


def test_ts_pipeline_with_stats_model():
    """ Tests PipelineTuner for time series forecasting task with AR model """
    train_data, test_data = get_ts_data(n_steps=200, forecast_length=5)

    ar_pipeline = Pipeline(PrimaryNode('ar'))

    # Tune AR model
    tuner_ar = PipelineTuner(pipeline=ar_pipeline, task=train_data.task, iterations=3)
    tuned_ar_pipeline = tuner_ar.tune_pipeline(input_data=train_data,
                                               loss_function=mse)

    is_tuning_finished = True

    assert is_tuning_finished
