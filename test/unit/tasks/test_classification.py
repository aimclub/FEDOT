import os

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, make_classification
from sklearn.metrics import roc_auc_score as roc_auc, f1_score as f1

from examples.simple.classification.image_classification_problem import run_image_classification_problem
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.integration.models.test_model import classification_dataset_with_redundant_features


def pipeline_simple() -> Pipeline:
    node_scaling = PipelineNode('scaling')
    node_svc = PipelineNode('svc', nodes_from=[node_scaling])
    node_lda = PipelineNode('lda', nodes_from=[node_scaling])
    node_final = PipelineNode('rf', nodes_from=[node_svc, node_lda])

    pipeline = Pipeline(node_final)

    return pipeline


def pipeline_with_pca() -> Pipeline:
    node_scaling = PipelineNode('scaling')
    node_pca = PipelineNode('pca', nodes_from=[node_scaling])
    node_lda = PipelineNode('lda', nodes_from=[node_scaling])
    node_final = PipelineNode('rf', nodes_from=[node_pca, node_lda])

    pipeline = Pipeline(node_final)

    return pipeline


def simple_text_pipeline() -> Pipeline:
    node_tfidf = PipelineNode('tfidf')
    model_node = PipelineNode('logit', nodes_from=[node_tfidf])
    pipeline = Pipeline(model_node)
    return pipeline


def get_synthetic_classification_data(n_samples=1000, n_features=10, random_state=None) -> InputData:
    synthetic_data = make_classification(n_samples=n_samples, n_features=n_features, random_state=random_state)
    input_data = InputData(idx=np.arange(0, len(synthetic_data[1])),
                           features=synthetic_data[0],
                           target=synthetic_data[1].reshape((-1, 1)),
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)

    return input_data


def get_iris_data() -> InputData:
    """ Prepare iris data for classification task in InputData format """
    synthetic_data = load_iris()
    input_data = InputData(idx=np.arange(0, len(synthetic_data.target)),
                           features=synthetic_data.data,
                           target=synthetic_data.target,
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table,
                           supplementary_data=SupplementaryData())
    return input_data


def get_classification_data(source: str, problem: str) -> InputData:
    test_file_path = str(os.path.dirname(__file__))
    if source == 'numpy':
        file = f'../../data/classification/{problem}_classification.npy'
        numpy_data = np.load(os.path.join(test_file_path, file))
        features_array = numpy_data[:, :-1]
        target_array = numpy_data[:, -1]
        return InputData.from_numpy(features_array=features_array,
                                    target_array=target_array)
    elif source == 'dataframe':
        file = f'../../data/classification/{problem}_classification.csv'
        df_data = pd.read_csv(os.path.join(test_file_path, file))
        features_df = df_data.iloc[:, :-1]
        target_df = df_data.iloc[:, -1]
        return InputData.from_dataframe(features_df=features_df,
                                        target_df=target_df)
    elif source == 'csv':
        file = f'../../data/classification/{problem}_classification.csv'
        return InputData.from_csv(
            os.path.join(test_file_path, file))


def get_image_classification_data(composite_flag: bool = True):
    """ Method for loading data with images in .npy format (training_data.npy, training_labels.npy,
    test_data.npy, test_labels.npy) that are used in tests.This npy files are a truncated version
    of the MNIST dataset, that contains only 10 first images.

    :param composite_flag: Flag that allows to run tests for complex composite models
    """
    test_data_path = '../../data/test_data.npy'
    test_labels_path = '../../data/test_labels.npy'
    train_data_path = '../../data/training_data.npy'
    train_labels_path = '../../data/training_labels.npy'

    test_file_path = str(os.path.dirname(__file__))
    training_path_features = os.path.join(test_file_path, train_data_path)
    training_path_labels = os.path.join(test_file_path, train_labels_path)
    test_path_features = os.path.join(test_file_path, test_data_path)
    test_path_labels = os.path.join(test_file_path, test_labels_path)

    roc_auc_on_valid, dataset_to_train, dataset_to_validate = run_image_classification_problem(
        train_dataset=(training_path_features,
                       training_path_labels),
        test_dataset=(test_path_features,
                      test_path_labels),
        composite_flag=composite_flag)

    return roc_auc_on_valid, dataset_to_train, dataset_to_validate


CLASSIFICATION_DATA_SOURCES = ['numpy',
                               'dataframe',
                               'csv',
                               # 'from_text_files',
                               # 'from_json_files',
                               ]


@pytest.mark.parametrize('source', CLASSIFICATION_DATA_SOURCES)
def test_binary_classification_pipeline_fit_correct(source: str):
    data = get_classification_data(source, 'simple')
    pipeline = pipeline_simple()
    train_data, test_data = train_test_data_setup(data, shuffle=True)

    pipeline.fit(input_data=train_data)
    results = pipeline.predict(input_data=test_data)

    roc_auc_on_test = roc_auc(y_true=test_data.target,
                              y_score=results.predict)

    assert roc_auc_on_test > 0.8


@pytest.mark.parametrize('source', CLASSIFICATION_DATA_SOURCES)
def test_multiclassification_pipeline_fit_correct(source: str):
    data = get_classification_data(source, 'multiclass')
    pipeline = pipeline_simple()
    train_data, test_data = train_test_data_setup(data, shuffle=True)

    pipeline.fit(input_data=train_data)
    results = pipeline.predict(input_data=test_data)

    roc_auc_on_test = roc_auc(y_true=test_data.target,
                              y_score=results.predict,
                              multi_class='ovr',  # TODO: strange bug when ovo is chosen
                              average='macro')

    assert roc_auc_on_test > 0.95


def test_classification_with_pca_pipeline_fit_correct():
    data = classification_dataset_with_redundant_features()
    pipeline_pca = pipeline_with_pca()
    pipeline = pipeline_simple()

    train_data, test_data = train_test_data_setup(data, shuffle=True)

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

    assert roc_auc_on_test_pca > roc_auc_on_test * 0.95 > 0.5  # add a small deviation


def test_output_mode_labels():
    data = get_iris_data()
    pipeline = pipeline_simple()
    train_data, test_data = train_test_data_setup(data, shuffle=True)

    pipeline.fit(input_data=train_data)
    results = pipeline.predict(input_data=test_data, output_mode='labels')
    results_probs = pipeline.predict(input_data=test_data)

    assert len(results.predict) == len(test_data.target)
    assert set(np.ravel(results.predict)) == {0, 1, 2}

    assert not np.array_equal(results_probs.predict, results.predict)


def test_output_mode_full_probs():
    data = get_classification_data('csv', 'simple')
    pipeline = pipeline_simple()
    train_data, test_data = train_test_data_setup(data, shuffle=True)

    pipeline.fit(input_data=train_data)
    results = pipeline.predict(input_data=test_data, output_mode='full_probs')
    results_default = pipeline.predict(input_data=test_data)
    results_probs = pipeline.predict(input_data=test_data, output_mode='probs')

    assert not np.array_equal(results_probs.predict, results.predict)
    assert np.array_equal(results_probs.predict, results_default.predict)
    assert results.predict.shape == (len(test_data.target), 2)
    assert results_probs.predict.shape == (len(test_data.target), 1)


def test_image_pipeline_fit_correct():
    roc_auc_on_valid, _, _ = get_image_classification_data()

    assert roc_auc_on_valid >= 0.5


def test_text_classification_pipeline_fit_correct():
    test_file_path = str(os.path.dirname(__file__))
    file = '../../data/simple_multimodal_classification_text.csv'
    data = InputData.from_csv(file_path=os.path.join(test_file_path, file),
                              data_type=DataTypesEnum.text)
    pipeline = simple_text_pipeline()
    train_data, test_data = train_test_data_setup(data, shuffle=True)

    pipeline.fit(input_data=train_data)
    results = pipeline.predict(input_data=test_data, output_mode='labels')

    f1_on_test = f1(y_true=test_data.target,
                    y_pred=results.predict,
                    average='micro')

    assert f1_on_test >= 0.5
