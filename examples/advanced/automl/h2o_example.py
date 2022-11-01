import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc, mean_squared_error, mean_absolute_error

from examples.advanced.time_series_forecasting.composing_pipelines import visualise
from examples.simple.pipeline_import_export import create_correct_path
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_types_repository import OperationTypesRepository
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
    node_lagged.parameters = {'window_size': window_size}
    node_root = SecondaryNode('h2o_regr', nodes_from=[node_lagged])

    pipeline = Pipeline(node_root)

    return pipeline


def export_h2o(pipeline, pipeline_path, test_data):
    # Export it
    pipeline.save(path=pipeline_path)

    # Import pipeline
    json_path_load = create_correct_path(pipeline_path)
    new_pipeline = Pipeline.from_serialized(json_path_load)

    results = new_pipeline.predict(input_data=test_data, output_mode="full_probs")
    prediction_after_export = results.predict[:, 0]
    print(f'After export {prediction_after_export[:4]}')


def h2o_classification_pipeline_evaluation():
    pipeline_path = "h2o_class"
    data = get_iris_data()
    pipeline = pipeline_h2o_class()
    train_data, test_data = train_test_data_setup(data, shuffle_flag=True)

    pipeline.fit(input_data=train_data)
    results = pipeline.predict(input_data=test_data, output_mode="full_probs")
    prediction_before_export = results.predict[:, 0]
    print(f'Before export {prediction_before_export[:4]}')
    roc_auc_on_test = roc_auc(y_true=test_data.target,
                              y_score=results.predict,
                              multi_class='ovo',
                              average='macro')
    #  H2o has troubles with serialization for now
    #  export_h2o(pipeline, pipeline_path, test_data)
    print(f"roc auc: {roc_auc_on_test}")


def h2o_regression_pipeline_evaluation():
    data = get_synthetic_regression_data()

    pipeline = pipeline_h2o_regr()
    train_data, test_data = train_test_data_setup(data)

    pipeline.fit(input_data=train_data)
    results = pipeline.predict(input_data=test_data)
    _, rmse_on_test = get_rmse_value(pipeline, train_data, test_data)
    print(f"RMSE {rmse_on_test}")


def h2o_ts_pipeline_evaluation(visualization=False):
    train_data, test_data = get_ts_data(n_steps=500, forecast_length=3)

    pipeline = pipeline_h2o_ts()
    pipeline.fit(input_data=train_data)
    test_pred = pipeline.predict(input_data=test_data)
    # Calculate metric
    test_pred = np.ravel(np.array(test_pred.predict))
    test_target = np.ravel(np.array(test_data.target))

    plot_info = [{'idx': train_data.idx,
                  'series': train_data.features,
                  'label': 'Actual time series'},
                 {'idx': test_data.idx,
                  'series': test_target,
                  'label': 'Test part'}]
    metrics_info = {}
    rmse = mean_squared_error(test_target, test_pred, squared=False)
    mae = mean_absolute_error(test_target, test_pred)

    metrics_info['Metrics'] = {'RMSE': round(rmse, 3),
                               'MAE': round(mae, 3)}
    if visualization:
        visualise(plot_info)
    print(metrics_info)


if __name__ == '__main__':
    with OperationTypesRepository.init_automl_repository() as _:
        h2o_classification_pipeline_evaluation()
    with OperationTypesRepository.init_automl_repository() as _:
        h2o_regression_pipeline_evaluation()
    with OperationTypesRepository.init_automl_repository() as _:
        h2o_ts_pipeline_evaluation(visualization=True)
