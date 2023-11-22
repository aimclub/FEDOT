import datetime
import random
from typing import Any

import numpy as np
import tensorflow as tf
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum
from hyperopt import hp
from sklearn.metrics import roc_auc_score as roc_auc

from examples.simple.classification.classification_pipelines import cnn_composite_pipeline
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.models.keras import check_input_array
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, ComplexityMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import set_random_seed

custom_search_space = {'gamma_filt': {
    'r': {
        'hyperopt-dist': hp.uniformint,
        'sampling-scope': [-254, 254],
        'type': 'discrete'},
    'g': {
        'hyperopt-dist': hp.uniformint,
        'sampling-scope': [-254, 254],
        'type': 'discrete'},
    'b': {
        'hyperopt-dist': hp.uniformint,
        'sampling-scope': [-254, 254],
        'type': 'discrete'},
    'ksize': {
        'hyperopt-dist': hp.uniformint,
        'sampling-scope': [0, 20],
        'type': 'discrete'}
}

}

def calculate_validation_metric(predicted: OutputData, dataset_to_validate: InputData) -> float:
    # the quality assessment for the simulation results
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                            y_score=predicted.predict,
                            multi_class="ovo")
    return roc_auc_value


def cnn_model_fit(idx: np.array, features: np.array, target: np.array, params: dict):
    # x_train, y_train = features, target
    # transformed_x_train, transform_flag = check_input_array(x_train)
    #
    # if transform_flag:
    #     print('Train data set was not scaled. The data was divided by 255.')
    #
    # if len(x_train.shape) == 3:
    #     transformed_x_train = np.expand_dims(x_train, -1)
    #
    # if len(target.shape) < 2:
    #     le = preprocessing.OneHotEncoder()
    #     y_train = le.fit_transform(y_train.reshape(-1, 1)).toarray()
    #
    # optimizer_params = {'loss': "categorical_crossentropy",
    #                     'optimizer': "adam",
    #                     'metrics': ["accuracy"]}
    #
    # model = tf.keras.Sequential(
    #     [
    #         tf.keras.layers.InputLayer(input_shape=[28, 28, 1]),
    #         tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    #         tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #         tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    #         tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #         tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
    #         tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dropout(0.5),
    #         tf.keras.layers.Dense(10, activation="softmax"),
    #     ])
    #
    # model.compile(**optimizer_params)
    # model.num_classes = 10

    # model.fit(transformed_x_train, y_train, batch_size=1, epochs=1,
    #           validation_split=0.1)
    model = None
    return model


#
#
def cnn_model_predict(fitted_model: Any, idx: np.array, features: np.array, params: dict):
    x_test = features
    transformed_x_test, transform_flag = check_input_array(x_test)

    if np.max(transformed_x_test) > 1:
        print('Test data set was not scaled. The data was divided by 255.')

    if len(x_test.shape) == 3:
        transformed_x_test = np.expand_dims(x_test, -1)

    # prediction = fitted_model.predict(transformed_x_test)

    prediction = np.asarray([[random.random()] for j in range(features.shape[0])])

    return prediction, 'table'


#

def preproc_predict(fitted_model: Any, idx: np.array, features: np.array, params: dict):
    # example of custom data pre-processing for predict state
    for i in range(features.shape[0]):
        features[i, :, :] = features[i, :, :] + np.random.normal(0, 30)
    return features, 'image'


def cnn_composite_pipeline(composite_flag: bool = True) -> Pipeline:
    """
    Returns pipeline with the following structure:

    .. image:: img_classification_pipelines/cnn_composite_pipeline.png
      :width: 55%

    Where cnn - convolutional neural network, rf - random forest

    :param composite_flag:  add additional random forest estimator
    """
    node_first = PipelineNode('gamma_filt')
    node_first.parameters = {'model_predict': preproc_predict}

    node_second = PipelineNode('cnn_1', nodes_from=[node_first])
    node_second.parameters = {'model_predict': cnn_model_predict,
                              'model_fit': cnn_model_fit}

    node_final = PipelineNode('rf', nodes_from=[node_second])

    pipeline = Pipeline(node_final)
    return pipeline


def run_image_classification_problem(train_dataset: tuple,
                                     test_dataset: tuple,
                                     composite_flag: bool = True):
    task = Task(TaskTypesEnum.classification)

    x_train, y_train = train_dataset[0], train_dataset[1]
    x_test, y_test = test_dataset[0], test_dataset[1]

    dataset_to_train = InputData.from_image(images=x_train,
                                            labels=y_train,
                                            task=task)
    dataset_to_validate = InputData.from_image(images=x_test,
                                               labels=y_test,
                                               task=task)

    dataset_to_train = dataset_to_train.subset_range(0, 100)

    initial_pipeline = cnn_composite_pipeline(composite_flag)

    # the choice of the metric for the pipeline quality assessment during composition
    quality_metric = ClassificationMetricsEnum.f1
    complexity_metric = ComplexityMetricsEnum.node_number
    metrics = [quality_metric, complexity_metric]
    # the choice and initialisation of the GP search
    composer_requirements = PipelineComposerRequirements(
        primary=['custom/preproc_image1', 'custom/preproc_image2'],
        secondary=['custom/cnn_1', 'custom/cnn_2'],
        timeout=datetime.timedelta(minutes=10),
        num_of_generations=20, n_jobs=1
    )

    pop_size = 5
    params = GPAlgorithmParameters(
        selection_types=[SelectionTypesEnum.spea2],
        genetic_scheme_type=GeneticSchemeTypesEnum.parameter_free,
        mutation_types=[MutationTypesEnum.single_change, parameter_change_mutation],
        pop_size=pop_size
    )

    # Create composer and with required composer params
    composer = (
        ComposerBuilder(task=task)
        .with_optimizer_params(params)
        .with_requirements(composer_requirements)
        .with_metrics(metrics)
        .with_initial_pipelines(initial_pipelines=[initial_pipeline] * pop_size)
        .build()
    )

    # the optimal pipeline generation by composition - the most time-consuming task
    pipeline_evo_composed = composer.compose_pipeline(data=dataset_to_train)[0]

    pipeline_evo_composed.show()
    print(pipeline_evo_composed.descriptive_id)

    pipeline_evo_composed.fit(input_data=dataset_to_train)

    # auto_model = Fedot(problem='classification', timeout=1, n_jobs=-1, preset='best_quality',
    #                    metric=['f1'], with_tuning=True, initial_assumption = pipeline,
    #                    available_models=[])
    #
    # auto_model.fit(features=dataset_to_train)

    # auto_model.predict(dataset_to_validate)
    # predictions = auto_model.prediction

    predictions = pipeline_evo_composed.predict(dataset_to_validate)

    roc_auc_on_valid = calculate_validation_metric(predictions,
                                                   dataset_to_validate)
    return roc_auc_on_valid, dataset_to_train, dataset_to_validate


if __name__ == '__main__':
    set_random_seed(1)

    training_set, testing_set = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    roc_auc_on_valid, dataset_to_train, dataset_to_validate = run_image_classification_problem(
        train_dataset=training_set,
        test_dataset=testing_set)

    print(roc_auc_on_valid)
