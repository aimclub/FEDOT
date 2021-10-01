import numpy as np

from examples.time_series.ts_forecasting_composing import display_validation_metric
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from sklearn.metrics import roc_auc_score as roc_auc
from test.unit.tasks.test_classification import get_iris_data
from test.unit.tasks.test_forecasting import get_ts_data
from test.unit.tasks.test_regression import get_synthetic_regression_data, get_rmse_value


def pipeline_h2o_class() -> Pipeline:
    node = PrimaryNode('h2o_class')
    pipeline = Pipeline(node)
    return pipeline


def pipeline_h2o_regr() -> Pipeline:
    node = PrimaryNode('h2o_regr')
    pipeline = Pipeline(node)
    return pipeline


def pipeline_h2o_ts(window_size: int = 20):
    node_lagged = PrimaryNode('lagged')
    node_lagged.custom_params = {'window_size': window_size}
    node_root = SecondaryNode('h2o_regr', nodes_from=[node_lagged])

    pipeline = Pipeline(node_root)

    return pipeline


def h2o_classification_pipeline_evaluation():
    data = get_iris_data()
    pipeline = pipeline_h2o_class()
    train_data, test_data = train_test_data_setup(data, shuffle_flag=True)

    pipeline.fit(input_data=train_data)
    results = pipeline.predict(input_data=test_data, output_mode="full_probs")

    roc_auc_on_test = roc_auc(y_true=test_data.target,
                              y_score=results.predict,
                              multi_class='ovo',
                              average='macro')

    print(f"roc auc: {roc_auc_on_test}")



def h2o_regression_pipeline_evaluation():
    data = get_synthetic_regression_data()

    pipeline = pipeline_h2o_regr()
    train_data, test_data = train_test_data_setup(data)

    pipeline.fit(input_data=train_data)
    _, rmse_on_test = get_rmse_value(pipeline, train_data, test_data)
    print(f"RMSE {rmse_on_test}")


def h2o_ts_pipeline_evaluation():
    train_data, test_data = get_ts_data(n_steps=500, forecast_length=3)

    pipeline = pipeline_h2o_ts()
    pipeline.fit(input_data=train_data)
    test_pred = pipeline.predict(input_data=test_data)

    # Calculate metric
    test_pred = np.ravel(np.array(test_pred.predict))
    test_target = np.ravel(np.array(test_data.target))
    display_validation_metric(predicted=test_pred,
                              real=test_target,
                              actual_values=test_data.features,
                              is_visualise=True)


if __name__ == '__main__':
    h2o_classification_pipeline_evaluation()
    h2o_regression_pipeline_evaluation()
    h2o_ts_pipeline_evaluation()
