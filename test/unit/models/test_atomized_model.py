import os
import json
import pytest

import numpy as np
from sklearn.metrics import mean_squared_error

from fedot.core.data.data import InputData
from fedot.core.operations.atomized_model import AtomizedModel
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utils import DEFAULT_PARAMS_STUB, fedot_project_root
from test.unit.utilities.test_pipeline_import_export import create_correct_path, create_func_delete_files


@pytest.fixture(scope='session', autouse=True)
def preprocessing_files_before_and_after_tests(request):
    paths = ['test_save_load_atomized_pipeline_correctly',
             'test_save_load_fitted_atomized_pipeline_correctly_loaded',
             'test_save_load_fitted_atomized_pipeline_correctly']

    delete_files = create_func_delete_files(paths)

    delete_files()
    request.addfinalizer(delete_files)


def create_pipeline() -> Pipeline:
    pipeline = Pipeline()
    node_logit = PrimaryNode('logit')

    node_lda = PrimaryNode('lda')
    node_lda.custom_params = {'solver': 'lsqr'}

    node_rf = SecondaryNode('rf')
    node_rf.nodes_from = [node_logit, node_lda]

    pipeline.add_node(node_rf)

    return pipeline


def create_atomized_model() -> AtomizedModel:
    """
    Example, how to create Atomized operation.
    """
    pipeline = create_pipeline()
    atomized_model = AtomizedModel(pipeline)

    return atomized_model


def create_atomized_model_with_several_atomized_models() -> AtomizedModel:
    pipeline = Pipeline()
    node_atomized_model_primary = PrimaryNode(operation_type=create_atomized_model())
    node_atomized_model_secondary = SecondaryNode(operation_type=create_atomized_model())
    node_atomized_model_secondary_second = SecondaryNode(operation_type=create_atomized_model())
    node_atomized_model_secondary_third = SecondaryNode(operation_type=create_atomized_model())

    node_atomized_model_secondary.nodes_from = [node_atomized_model_primary]
    node_atomized_model_secondary_second.nodes_from = [node_atomized_model_primary]
    node_atomized_model_secondary_third.nodes_from = [node_atomized_model_secondary,
                                                      node_atomized_model_secondary_second]

    pipeline.add_node(node_atomized_model_secondary_third)
    atomized_model = AtomizedModel(pipeline)

    return atomized_model


def create_pipeline_with_several_nested_atomized_model() -> Pipeline:
    pipeline = Pipeline()
    atomized_op = create_atomized_model_with_several_atomized_models()
    node_atomized_model = PrimaryNode(operation_type=atomized_op)

    node_atomized_model_secondary = SecondaryNode(operation_type=create_atomized_model())
    node_atomized_model_secondary.nodes_from = [node_atomized_model]

    node_knn = SecondaryNode('knn')
    node_knn.custom_params = {'n_neighbors': 9}
    node_knn.nodes_from = [node_atomized_model]

    node_knn_second = SecondaryNode('knn')
    node_knn_second.custom_params = {'n_neighbors': 5}
    node_knn_second.nodes_from = [node_atomized_model, node_atomized_model_secondary, node_knn]

    node_atomized_model_secondary_second = \
        SecondaryNode(operation_type=create_atomized_model_with_several_atomized_models())

    node_atomized_model_secondary_second.nodes_from = [node_knn_second]

    pipeline.add_node(node_atomized_model_secondary_second)
    return pipeline


def create_input_data():
    train_file_path = os.path.join('test', 'data', 'scoring', 'scoring_train.csv')
    test_file_path = os.path.join('test', 'data', 'scoring', 'scoring_test.csv')
    full_train_file_path = os.path.join(str(fedot_project_root()), train_file_path)
    full_test_file_path = os.path.join(str(fedot_project_root()), test_file_path)

    train_data = InputData.from_csv(full_train_file_path)
    test_data = InputData.from_csv(full_test_file_path)

    return train_data, test_data


def test_save_load_atomized_pipeline_correctly():
    pipeline = create_pipeline_with_several_nested_atomized_model()

    json_actual, _ = pipeline.save('test_save_load_atomized_pipeline_correctly')

    json_path_load = create_correct_path('test_save_load_atomized_pipeline_correctly')

    with open(json_path_load, 'r') as json_file:
        json_expected = json.load(json_file)

    pipeline_loaded = Pipeline()
    pipeline_loaded.load(json_path_load)

    assert pipeline.length == pipeline_loaded.length
    assert json_actual == json.dumps(json_expected, indent=4)


def test_save_load_fitted_atomized_pipeline_correctly():
    train_data, test_data = create_input_data()

    pipeline = create_pipeline_with_several_nested_atomized_model()

    pipeline.fit(train_data)
    before_save_predicted = pipeline.predict(test_data)
    json_actual, _ = pipeline.save('test_save_load_fitted_atomized_pipeline_correctly')

    json_path_load = create_correct_path('test_save_load_fitted_atomized_pipeline_correctly')

    pipeline_loaded = Pipeline()
    pipeline_loaded.load(json_path_load)
    json_expected, _ = pipeline_loaded.save('test_save_load_fitted_atomized_pipeline_correctly_loaded')

    assert pipeline.length == pipeline_loaded.length
    assert json_actual == json_expected

    pipeline_loaded.fit_from_scratch(train_data)
    after_save_predicted = pipeline_loaded.predict(test_data)

    bfr_save_mse = mean_squared_error(y_true=test_data.target, y_pred=before_save_predicted.predict)
    aft_load_mse = mean_squared_error(y_true=test_data.target, y_pred=after_save_predicted.predict)

    assert np.isclose(aft_load_mse, bfr_save_mse)


def test_fit_predict_atomized_model_correctly():
    """ Check if pipeline nested in AtomizedModel can give the same result as Pipeline stand-alone """
    train_data, test_data = create_input_data()

    pipeline = create_pipeline_with_several_nested_atomized_model()
    atomized_model = AtomizedModel(pipeline)

    pipeline.fit(train_data)
    predicted_values = pipeline.predict(test_data)

    pipeline.unfit()

    fitted_atomized_model = pipeline.fit(train_data)
    predicted_atomized_output = pipeline.predict(test_data)
    predicted_atomized_values = predicted_atomized_output.predict

    source_mse = mean_squared_error(y_true=test_data.target, y_pred=predicted_values.predict)
    atomized_mse = mean_squared_error(y_true=test_data.target, y_pred=predicted_atomized_values)

    assert np.isclose(atomized_mse, source_mse)


def test_create_empty_atomized_model_raised_exception():
    with pytest.raises(Exception):
        empty_pipeline = Pipeline()
        AtomizedModel(empty_pipeline)


def test_fine_tune_atomized_model_correct():
    train_data, test_data = create_input_data()

    atm_model = create_atomized_model()
    dummy_atomized_model = create_atomized_model()

    fine_tuned_atomized_model = atm_model.fine_tune(loss_function=mean_squared_error,
                                                    input_data=train_data,
                                                    iterations=5,
                                                    timeout=1)
    dummy_atomized_model.fit(DEFAULT_PARAMS_STUB, train_data)

    fitted_dummy_model, _ = dummy_atomized_model.fit(DEFAULT_PARAMS_STUB, train_data)
    fitted_fine_tuned_atomized_model, _ = fine_tuned_atomized_model.fit(DEFAULT_PARAMS_STUB, train_data)

    after_tuning_output = fine_tuned_atomized_model.predict(fitted_fine_tuned_atomized_model, data=test_data)
    after_tuning_predicted = after_tuning_output.predict
    before_tuning_output = dummy_atomized_model.predict(fitted_dummy_model, data=test_data)
    before_tuning_predicted = before_tuning_output.predict

    aft_tun_mse = mean_squared_error(y_true=test_data.target, y_pred=after_tuning_predicted)
    bfr_tun_mse = mean_squared_error(y_true=test_data.target, y_pred=before_tuning_predicted)

    deviation = 0.50 * bfr_tun_mse
    assert aft_tun_mse <= (bfr_tun_mse + deviation)
