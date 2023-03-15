import logging
import os
import random
from functools import partial

import numpy as np
from sklearn.metrics import mean_squared_error

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.integration.composer.test_composer import to_categorical_codes


def get_regression_pipeline():
    first = PipelineNode(operation_type='scaling')
    final = PipelineNode(operation_type='ridge',
                         nodes_from=[first])

    pipeline = Pipeline(final)
    return pipeline


def get_regression_data():
    test_file_path = str(os.path.dirname(__file__))
    file = '../../data/simple_regression_train.csv'
    input_data = InputData.from_csv(
        os.path.join(test_file_path, file), task=Task(TaskTypesEnum.regression))
    input_data.idx = to_categorical_codes(categorical_ids=input_data.idx)
    return input_data


def custom_metric_for_synt_data(pipeline, fit_data, predict_data, **kwargs):
    np.random.seed(42)
    random.seed(42)

    # reference_data is added for compatibility with composer interface
    pipeline.fit_from_scratch(fit_data)
    results = pipeline.predict(predict_data)
    error = mean_squared_error(y_true=results.target,
                               y_pred=results.predict, squared=False)
    return error


def test_synthetic_metric():
    """
    This supplementary test checks that prediction of optimal pipeline is identical to the target,
    so the metric is equals to 0.
    """
    ref_pipeline = get_regression_pipeline()
    input_data = get_regression_data()

    train_data, test_data = train_test_data_setup(input_data)

    ref_pipeline.fit(train_data)
    ground_truth = ref_pipeline.predict(test_data)
    test_data.target = ground_truth.predict

    metric_func = partial(custom_metric_for_synt_data, fit_data=train_data, predict_data=test_data)

    assert metric_func(ref_pipeline) == 0.0


def test_synthetic_regression_automl():
    """
    This test compares that pre-known optimal pipeline is can be found by composer.
    If correct, the best fitness should be close to 0.
    """

    ref_pipeline = get_regression_pipeline()
    input_data = get_regression_data()

    train_data, test_data = train_test_data_setup(input_data)

    metric_func = partial(custom_metric_for_synt_data, fit_data=train_data, predict_data=test_data)

    # generate synthetic target
    ref_pipeline.fit(train_data)
    ground_truth = ref_pipeline.predict(test_data)
    test_data.target = ground_truth.predict

    # run automl
    auto_model = Fedot(problem='regression', logging_level=logging.ERROR, timeout=1,
                       metric=metric_func,
                       cv_folds=None,
                       available_operations=['scaling',
                                             'normalization',
                                             'pca',
                                             'ridge',
                                             'linear'],
                       preset='best_quality',
                       with_tuning=False)
    auto_model.fit(train_data)

    assert min(auto_model.history.historical_fitness[-1]) < 0.01
