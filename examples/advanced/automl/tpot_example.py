import numpy as np

from examples.advanced.time_series_forecasting.composing_pipelines import visualise
from examples.simple.pipeline_import_export import create_correct_path
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from sklearn.metrics import roc_auc_score as roc_auc, mean_squared_error, mean_absolute_error

from fedot.core.repository.operation_types_repository import OperationTypesRepository
from test.unit.tasks.test_classification import get_iris_data
from test.unit.tasks.test_forecasting import get_ts_data
from test.unit.tasks.test_regression import get_synthetic_regression_data, get_rmse_value


def pipeline_tpot_class() -> Pipeline:
    node = PrimaryNode('tpot_class')
    pipeline = Pipeline(node)

    return pipeline


def pipeline_tpot_regr() -> Pipeline:
    node = PrimaryNode('tpot_regr')
    pipeline = Pipeline(node)

    return pipeline


def pipeline_tpot_ts(window_size: int = 20):
    node_lagged = PrimaryNode('lagged')
    node_lagged.parameters = {'window_size': window_size}
    node_root = SecondaryNode('tpot_regr', nodes_from=[node_lagged])

    pipeline = Pipeline(node_root)

    return pipeline


def tpot_classification_pipeline_evaluation():
    pipeline_path = "tpot_class"
    data = get_iris_data()
    pipeline = pipeline_tpot_class()
    train_data, test_data = train_test_data_setup(data, shuffle_flag=True)

    pipeline.fit(input_data=train_data)
    results = pipeline.predict(input_data=test_data, output_mode="full_probs")
    prediction_before_export = results.predict[:, 0]
    print(f'Before export {prediction_before_export[:4]}')

    # Export it
    pipeline.save(path=pipeline_path)

    # Import pipeline
    json_path_load = create_correct_path(pipeline_path)
    new_pipeline = Pipeline.from_serialized(json_path_load)

    predicted_output_after_export = new_pipeline.predict(test_data, output_mode="full_probs")
    prediction_after_export = predicted_output_after_export.predict[:, 0]

    print(f'After import {prediction_after_export[:4]}')

    roc_auc_on_test = roc_auc(y_true=test_data.target,
                              y_score=results.predict,
                              multi_class='ovo',
                              average='macro')
    print(f"roc_auc {roc_auc_on_test}")


def tpot_regression_pipeline_evaluation():
    pipeline_path = "tpot_regr"
    data = get_synthetic_regression_data()

    pipeline = pipeline_tpot_regr()
    train_data, test_data = train_test_data_setup(data)

    pipeline.fit(input_data=train_data)

    results = pipeline.predict(test_data)
    print(f'Before export {results.predict[:4]}')

    # Export it
    pipeline.save(path=pipeline_path)

    # Import pipeline
    json_path_load = create_correct_path(pipeline_path)
    new_pipeline = Pipeline.from_serialized(json_path_load)

    predicted_output_after_export = new_pipeline.predict(test_data)
    prediction_after_export = predicted_output_after_export.predict[:4]

    print(f'After import {prediction_after_export[:4]}')

    _, rmse_on_test = get_rmse_value(pipeline, train_data, test_data)
    print(f"RMSE {rmse_on_test}")


def tpot_ts_pipeline_evaluation():
    pipeline_path = "tpot_ts"
    train_data, test_data = get_ts_data(n_steps=500, forecast_length=3)

    pipeline = pipeline_tpot_ts()
    pipeline.fit(input_data=train_data)
    test_pred = pipeline.predict(input_data=test_data)

    print(f'Before export {test_pred.predict[:4]}')

    # Export it
    pipeline.save(path=pipeline_path)

    # Import pipeline
    json_path_load = create_correct_path(pipeline_path)
    new_pipeline = Pipeline.from_serialized(json_path_load)

    predicted_output_after_export = new_pipeline.predict(test_data)
    prediction_after_export = predicted_output_after_export.predict[:4]

    print(f'After import {prediction_after_export[:4]}')

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
    visualise(plot_info)
    print(metrics_info)


if __name__ == '__main__':
    with OperationTypesRepository.init_automl_repository() as _:
        tpot_classification_pipeline_evaluation()
        tpot_regression_pipeline_evaluation()
        tpot_ts_pipeline_evaluation()
