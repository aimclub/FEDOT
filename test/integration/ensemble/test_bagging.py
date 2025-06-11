import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import f1_score, r2_score

from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
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
