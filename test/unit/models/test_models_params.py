import numpy as np

from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.synth_dataset_generator import classification_dataset, regression_dataset


def get_knn_reg_pipeline(k_neighbors):
    """ Function return pipeline with K-nn regression model in it """
    node_scaling = PrimaryNode('scaling')
    node_final = SecondaryNode('knnreg', nodes_from=[node_scaling])
    node_final.parameters = {'n_neighbors': k_neighbors}
    pipeline = Pipeline(node_final)
    return pipeline


def get_knn_class_pipeline(k_neighbors):
    """ Function return pipeline with K-nn classification model in it """
    node_scaling = PrimaryNode('scaling')
    node_final = SecondaryNode('knn', nodes_from=[node_scaling])
    node_final.parameters = {'n_neighbors': k_neighbors}
    pipeline = Pipeline(node_final)
    return pipeline


def test_knn_reg_with_invalid_params_fit_correctly():
    """ The function define a pipeline with incorrect parameters in the K-nn regression
    model. During the training of the pipeline, the parameter 'n_neighbors' is corrected
    """
    samples_amount = 100
    k_neighbors = 150

    features_options = {'informative': 1, 'bias': 0.0}
    features_amount = 3
    x_data, y_data = regression_dataset(samples_amount=samples_amount,
                                        features_amount=features_amount,
                                        features_options=features_options,
                                        n_targets=1,
                                        noise=0.0, shuffle=True)

    # Define regression task
    task = Task(TaskTypesEnum.regression)

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(x_data)), features=x_data,
                            target=y_data, task=task, data_type=DataTypesEnum.table)

    # Prepare regression pipeline
    pipeline = get_knn_reg_pipeline(k_neighbors)

    # Fit it
    pipeline.fit(train_input)

    is_pipeline_was_fitted = True
    assert is_pipeline_was_fitted


def test_knn_class_with_invalid_params_fit_correctly():
    """ The function define a pipeline with incorrect parameters in the K-nn classification
    model. During the training of the pipeline, the parameter 'n_neighbors' is corrected
    """

    samples_amount = 100
    k_neighbors = 150

    features_options = {'informative': 1, 'redundant': 0,
                        'repeated': 0, 'clusters_per_class': 1}
    x_data, y_data = classification_dataset(samples_amount=samples_amount,
                                            features_amount=3,
                                            classes_amount=2,
                                            features_options=features_options)

    # Define regression task
    task = Task(TaskTypesEnum.classification)

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(x_data)), features=x_data,
                            target=y_data, task=task, data_type=DataTypesEnum.table)

    # Prepare classification pipeline
    pipeline = get_knn_class_pipeline(k_neighbors)

    # Fit it
    pipeline.fit(train_input)

    is_pipeline_was_fitted = True
    assert is_pipeline_was_fitted
