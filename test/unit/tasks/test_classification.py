import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.evaluation.operation_implementations.models.keras import CustomCNNImplementation, \
    check_input_array, create_deep_cnn, fit_cnn, predict_cnn
from data.data_manager import classification_dataset_with_redunant_features, get_iris_data,\
    get_binary_classification_data, get_image_classification_data
from data.pipeline_manager import pipeline_simple, pipeline_with_pca


def test_multiclassification_pipeline_fit_correct():
    data = get_iris_data()
    pipeline = pipeline_simple()
    train_data, test_data = train_test_data_setup(data, shuffle_flag=True)

    pipeline.fit(input_data=train_data)
    results = pipeline.predict(input_data=test_data)

    roc_auc_on_test = roc_auc(y_true=test_data.target,
                              y_score=results.predict,
                              multi_class='ovo',
                              average='macro')

    assert roc_auc_on_test > 0.95


def test_classification_with_pca_pipeline_fit_correct():
    data = classification_dataset_with_redunant_features()
    pipeline_pca = pipeline_with_pca()
    pipeline = pipeline_simple()

    train_data, test_data = train_test_data_setup(data, shuffle_flag=True)

    pipeline.fit(input_data=train_data)
    pipeline_pca.fit(input_data=train_data)

    results = pipeline.predict(input_data=test_data)
    results_pca = pipeline_pca.predict(input_data=test_data)

    roc_auc_on_test = roc_auc(y_true=test_data.target,
                              y_score=results.predict,
                              multi_class='ovo',
                              average='macro')

    roc_auc_on_test_pca = roc_auc(y_true=test_data.target,
                                  y_score=results_pca.predict,
                                  multi_class='ovo',
                                  average='macro')

    assert roc_auc_on_test_pca > roc_auc_on_test > 0.5


def test_output_mode_labels():
    data = get_iris_data()
    pipeline = pipeline_simple()
    train_data, test_data = train_test_data_setup(data, shuffle_flag=True)

    pipeline.fit(input_data=train_data)
    results = pipeline.predict(input_data=test_data, output_mode='labels')
    results_probs = pipeline.predict(input_data=test_data)

    assert len(results.predict) == len(test_data.target)
    assert set(results.predict) == {0, 1, 2}

    assert not np.array_equal(results_probs.predict, results.predict)


def test_output_mode_full_probs():
    data = get_binary_classification_data()
    pipeline = pipeline_simple()
    train_data, test_data = train_test_data_setup(data, shuffle_flag=True)

    pipeline.fit(input_data=train_data)
    results = pipeline.predict(input_data=test_data, output_mode='full_probs')
    results_default = pipeline.predict(input_data=test_data)
    results_probs = pipeline.predict(input_data=test_data, output_mode='probs')

    assert not np.array_equal(results_probs.predict, results.predict)
    assert np.array_equal(results_probs.predict, results_default.predict)
    assert results.predict.shape == (len(test_data.target), 2)
    assert results_probs.predict.shape == (len(test_data.target),)


def test_image_classification_quality():
    roc_auc_on_valid, _, _ = get_image_classification_data()
    deviation_composite = roc_auc_on_valid - 0.5

    roc_auc_on_valid, _, _ = get_image_classification_data(composite_flag=False)
    deviation_simple = roc_auc_on_valid - 0.5

    assert abs(deviation_composite) < 0.25
    assert abs(deviation_simple) < 0.35


def test_cnn_custom_class():
    cnn_class = CustomCNNImplementation()

    assert type(cnn_class.model) == tf.keras.Sequential
    assert type(cnn_class) == CustomCNNImplementation


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
    assert transform_flag == True
    assert cnn_model.input_shape[1:] == image_shape
    assert cnn_model.output_shape[1] == num_classes
    assert type(prediction) == np.ndarray
