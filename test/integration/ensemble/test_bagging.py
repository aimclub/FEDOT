import datetime

import hyperopt
from hyperopt import hp
import numpy as np
from golem.core.tuning.simultaneous import SimultaneousTuner
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import f1_score, r2_score

from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.metrics_repository import RegressionMetricsEnum, ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.data.data_split import train_test_data_setup


#================== Test Data Preparation ========================

def get_multiclass_data():
    """Generate multiclass classification data"""
    X, y = make_classification(n_samples=100, n_features=5,
                              n_classes=3, n_informative=3, random_state=42)
    return InputData(idx=np.arange(0, len(X)), features=X, target=y,
                     data_type=DataTypesEnum.table,
                     task=Task(TaskTypesEnum.classification))

def get_regression_data():
    """Generate regression data"""
    X, y = make_regression(n_samples=100, n_features=5,
                          n_informative=3, random_state=42)
    return InputData(idx=np.arange(0, len(X)), features=X, target=y,
                     data_type=DataTypesEnum.table,
                     task=Task(TaskTypesEnum.regression))

#================== Bagging ========================

def test_bagging_classification():
    """Single-model case processing test in blending"""
    input_data = get_multiclass_data()
    train, test = train_test_data_setup(input_data)

    models = ['cb_bag', 'xgb_bag', 'lgbm_bag']

    # Pipeline with single branch
    for model in models:
        pipeline = PipelineBuilder().add_node(model).build()

        pipeline.fit(train)
        # Checking predictions
        predictions = pipeline.predict(test, output_mode='labels')
        assert len(predictions.predict) == len(test.target)

        # Checking metrics
        f1 = f1_score(predictions.target, predictions.predict, average='macro')
        assert f1 > 0.5  # better than constant predictor

def test_bagging_regression():
    """Single-model case processing test in blending"""
    input_data = get_regression_data()
    train, test = train_test_data_setup(input_data)

    models = ['cbreg_bag', 'xgbreg_bag', 'lgbmreg_bag']

    # Pipeline with single branch
    for model in models:
        pipeline = PipelineBuilder().add_node(model).build()

        pipeline.fit(train)
        # Checking predictions
        predictions = pipeline.predict(test)
        assert len(predictions.predict) == len(test.target)

        # Checking metrics
        r2 = r2_score(predictions.target, predictions.predict)
        assert r2 > 0  # better than constant predictor

def test_bagging_reg_tuning_correctness():
    """Test bagging hyperparameters tuning correctness"""
    # Input data
    input_data = get_regression_data()
    train_data, test = train_test_data_setup(input_data)

    # Constants
    model = 'lgbmreg_bag'
    task = Task(TaskTypesEnum.regression)
    tuner = SimultaneousTuner
    metric = RegressionMetricsEnum.MSE
    timeout = datetime.timedelta(minutes=0.5)
    algo = hyperopt.rand.suggest

    # Tuner
    pipeline_tuner = TunerBuilder(task) \
        .with_tuner(tuner) \
        .with_metric(metric) \
        .with_timeout(timeout) \
        .with_additional_params(algo=algo) \
        .build(train_data)

    # Pipeline with single branch
    pipeline = PipelineBuilder().add_node(model).build()
    pipeline_tuner.tune(pipeline)
    assert pipeline_tuner.was_tuned == True

def test_bagging_clf_tuning_correctness():
    """Test bagging hyperparameters tuning correctness"""
    # Input data
    input_data = get_multiclass_data()
    train_data, test = train_test_data_setup(input_data)

    # Constants
    model = 'xgb_bag'
    task = Task(TaskTypesEnum.classification)
    tuner = SimultaneousTuner
    metric = ClassificationMetricsEnum.ROCAUC
    timeout = datetime.timedelta(minutes=0.5)
    algo = hyperopt.rand.suggest

    # Tuner
    pipeline_tuner = TunerBuilder(task) \
        .with_tuner(tuner) \
        .with_metric(metric) \
        .with_timeout(timeout) \
        .with_additional_params(algo=algo) \
        .build(train_data)

    # Pipeline with single branch
    pipeline = PipelineBuilder().add_node(model).build()
    pipeline_tuner.tune(pipeline)
    assert pipeline_tuner.was_tuned == True
