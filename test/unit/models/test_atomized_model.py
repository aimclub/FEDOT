import json

import pytest
from sklearn.metrics import mean_squared_error

from fedot.core.operations.atomized_model import AtomizedModel
from fedot.core.pipelines.pipeline import Pipeline
from data.pipeline_manager import create_pipeline_with_several_nested_atomized_model, create_atomized_model
from data.data_manager import create_data_for_train
from data.utils import create_correct_path, create_func_delete_files


@pytest.fixture(scope='session', autouse=True)
def preprocessing_files_before_and_after_tests(request):
    paths = ['test_save_load_atomized_pipeline_correctly',
             'test_save_load_fitted_atomized_pipeline_correctly_loaded',
             'test_save_load_fitted_atomized_pipeline_correctly']

    delete_files = create_func_delete_files(paths)

    delete_files()
    request.addfinalizer(delete_files)


def test_save_load_atomized_pipeline_correctly():
    pipeline = create_pipeline_with_several_nested_atomized_model()

    json_actual = pipeline.save('test_save_load_atomized_pipeline_correctly')

    json_path_load = create_correct_path('test_save_load_atomized_pipeline_correctly')

    with open(json_path_load, 'r') as json_file:
        json_expected = json.load(json_file)

    pipeline_loaded = Pipeline()
    pipeline_loaded.load(json_path_load)

    assert pipeline.length == pipeline_loaded.length
    assert json_actual == json.dumps(json_expected, indent=4)


def test_save_load_fitted_atomized_pipeline_correctly():
    pipeline = create_pipeline_with_several_nested_atomized_model()

    train_data, test_data = create_data_for_train()
    pipeline.fit(train_data)

    json_actual = pipeline.save('test_save_load_fitted_atomized_pipeline_correctly')

    json_path_load = create_correct_path('test_save_load_fitted_atomized_pipeline_correctly')

    pipeline_loaded = Pipeline()
    pipeline_loaded.load(json_path_load)
    json_expected = pipeline_loaded.save('test_save_load_fitted_atomized_pipeline_correctly_loaded')

    assert pipeline.length == pipeline_loaded.length
    assert json_actual == json_expected

    before_save_predicted = pipeline.predict(test_data)

    pipeline_loaded.fit(train_data)
    after_save_predicted = pipeline_loaded.predict(test_data)

    bfr_tun_mse = mean_squared_error(y_true=test_data.target, y_pred=before_save_predicted.predict)
    aft_tun_mse = mean_squared_error(y_true=test_data.target, y_pred=after_save_predicted.predict)

    assert aft_tun_mse <= bfr_tun_mse


def test_fit_predict_atomized_model_correctly():
    train_data, test_data = create_data_for_train()

    pipeline = create_pipeline_with_several_nested_atomized_model()
    atomized_model = AtomizedModel(pipeline)

    pipeline.fit(train_data)
    predicted_values = pipeline.predict(test_data)

    fitted_atomized_model, _ = atomized_model.fit(train_data)
    predicted_atomized_output = atomized_model.predict(fitted_atomized_model, test_data)
    predicted_atomized_values = predicted_atomized_output.predict

    bfr_tun_mse = mean_squared_error(y_true=test_data.target, y_pred=predicted_values.predict)
    aft_tun_mse = mean_squared_error(y_true=test_data.target, y_pred=predicted_atomized_values)

    assert aft_tun_mse == bfr_tun_mse


def test_create_empty_atomized_model_raised_exception():
    with pytest.raises(Exception):
        empty_pipeline = Pipeline()
        AtomizedModel(empty_pipeline)


def test_fine_tune_atomized_model_correct():
    train_data, test_data = create_data_for_train()

    atm_model = create_atomized_model()
    dummy_atomized_model = create_atomized_model()

    fine_tuned_atomized_model = atm_model.fine_tune(loss_function=mean_squared_error,
                                                    input_data=train_data,
                                                    iterations=5,
                                                    timeout=1)
    dummy_atomized_model.fit(train_data)

    fitted_dummy_model, _ = dummy_atomized_model.fit(train_data)
    fitted_fine_tuned_atomized_model, _ = fine_tuned_atomized_model.fit(train_data)

    after_tuning_output = fine_tuned_atomized_model.predict(fitted_fine_tuned_atomized_model, data=test_data)
    after_tuning_predicted = after_tuning_output.predict
    before_tuning_output = dummy_atomized_model.predict(fitted_dummy_model, data=test_data)
    before_tuning_predicted = before_tuning_output.predict

    aft_tun_mse = mean_squared_error(y_true=test_data.target, y_pred=after_tuning_predicted)
    bfr_tun_mse = mean_squared_error(y_true=test_data.target, y_pred=before_tuning_predicted)

    deviation = 0.50 * bfr_tun_mse
    assert aft_tun_mse <= (bfr_tun_mse + deviation)
