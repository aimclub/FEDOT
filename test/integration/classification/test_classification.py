import numpy as np

try:
    import tensorflow as tf
except ModuleNotFoundError:
    from golem.utilities.requirements_notificator import warn_requirement

    warn_requirement('tensorflow')

from test.unit.common_tests import is_predict_ignores_target
from test.unit.tasks.test_classification import get_image_classification_data

from fedot.core.operations.evaluation.operation_implementations.models.keras import (
    FedotCNNImplementation,
    check_input_array,
    create_deep_cnn,
    fit_cnn,
    predict_cnn
)


def check_predict_cnn_correct(model, dataset_to_validate):
    return is_predict_ignores_target(
        predict_func=predict_cnn,
        predict_args={'trained_model': model},
        data_arg_name='predict_data',
        input_data=dataset_to_validate,
    )


def test_cnn_custom_class():
    cnn_class = FedotCNNImplementation()

    assert cnn_class.params is not None
    assert type(cnn_class) == FedotCNNImplementation


def test_image_classification_quality():
    roc_auc_on_valid, _, _ = get_image_classification_data()
    deviation_composite = roc_auc_on_valid - 0.5

    roc_auc_on_valid, _, _ = get_image_classification_data(composite_flag=False)
    deviation_simple = roc_auc_on_valid - 0.5

    assert abs(deviation_composite) < 0.25
    assert abs(deviation_simple) < 0.35


def test_cnn_methods():
    _, dataset_to_train, dataset_to_validate = get_image_classification_data()
    image_shape = (28, 28, 1)
    num_classes = 7
    epochs = 10
    batch_size = 128

    cnn_model = create_deep_cnn(input_shape=image_shape,
                                num_classes=num_classes)

    transformed_x_train, transform_flag = check_input_array(x_train=dataset_to_train.features)

    model = fit_cnn(train_data=dataset_to_train,
                    model=cnn_model,
                    epochs=epochs,
                    batch_size=batch_size)

    prediction = predict_cnn(trained_model=model,
                             predict_data=dataset_to_validate)

    assert type(cnn_model) == tf.keras.Sequential
    assert transform_flag is True
    assert cnn_model.input_shape[1:] == image_shape
    assert cnn_model.output_shape[1] == num_classes
    assert type(prediction) == np.ndarray
    assert check_predict_cnn_correct(model, dataset_to_validate)
