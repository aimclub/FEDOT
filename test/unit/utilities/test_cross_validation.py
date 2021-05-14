import numpy as np
from datetime import timedelta

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.operations.cross_validation import cross_validation
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.composer.gp_composer.gp_composer import GPComposerRequirements, GPComposerBuilder


def classification_dataset():
    samples = 1000
    x = 10.0 * np.random.rand(samples, ) - 5.0
    x = np.expand_dims(x, axis=1)
    y = 1.0 / (1.0 + np.exp(np.power(x, -1.0)))
    threshold = 0.5
    classes = np.array([0.0 if val <= threshold else 1.0 for val in y])
    classes = np.expand_dims(classes, axis=1)
    data = InputData(features=x, target=classes, idx=np.arange(0, len(x)),
                     task=Task(TaskTypesEnum.classification),
                     data_type=DataTypesEnum.table)

    return data


def sample_chain():
    return Chain(SecondaryNode(operation_type='logit',
                               nodes_from=[PrimaryNode(operation_type='xgboost'),
                                           PrimaryNode(operation_type='scaling')]))


def test_cv_metric_correct():
    source = classification_dataset()
    chain = sample_chain()

    actual_value = cross_validation(chain=chain, reference_data=source, cv=10,
                                    metrics=[ClassificationMetricsEnum.ROCAUC_penalty,
                                             ClassificationMetricsEnum.accuracy,
                                             ClassificationMetricsEnum.logloss])

    assert all(list(map(lambda x: x >= -1, actual_value)))


def test_cv_with_composer_optimisation_correct():
    task = Task(task_type=TaskTypesEnum.classification)
    dataset_to_compose = classification_dataset()

    models_repo = OperationTypesRepository()
    available_model_types, _ = models_repo.suitable_operation(task_type=task.task_type, tags=['simple'])

    metric_function = [ClassificationMetricsEnum.ROCAUC_penalty,
                       ClassificationMetricsEnum.accuracy,
                       ClassificationMetricsEnum.logloss]

    composer_requirements = GPComposerRequirements(primary=available_model_types,
                                                   secondary=available_model_types,
                                                   max_lead_time=timedelta(minutes=2),
                                                   num_of_generations=5)

    builder = GPComposerBuilder(task).with_requirements(composer_requirements).with_metrics(metric_function)
    composer = builder.build()

    chain_evo_composed = composer.compose_chain(data=dataset_to_compose, is_visualise=False, folds=4)[0]

    assert isinstance(chain_evo_composed, Chain)
