import pytest
from datetime import timedelta

from sklearn.metrics import roc_auc_score as roc_auc

from fedot.api.main import Fedot
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.operations.cross_validation import cross_validation
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, ClusteringMetricsEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.composer.gp_composer.gp_composer import GPComposerRequirements, GPComposerBuilder
from cases.credit_scoring.credit_scoring_problem import get_scoring_data
from test.unit.models.test_model import classification_dataset


def sample_chain():
    return Chain(SecondaryNode(operation_type='logit',
                               nodes_from=[PrimaryNode(operation_type='xgboost'),
                                           PrimaryNode(operation_type='scaling')]))


def get_data(task):
    full_path_train, full_path_test = get_scoring_data()
    dataset_to_compose = InputData.from_csv(full_path_train, task=task)
    dataset_to_validate = InputData.from_csv(full_path_test, task=task)

    return dataset_to_compose, dataset_to_validate


def test_cv_multiple_metrics_evaluated_correct(classification_dataset):
    chain = sample_chain()

    actual_value = cross_validation(chain=chain, reference_data=classification_dataset, cv_folds=10,
                                    metrics=[ClassificationMetricsEnum.ROCAUC_penalty,
                                             ClassificationMetricsEnum.accuracy,
                                             ClassificationMetricsEnum.logloss])
    all_metrics_correct = all(list(map(lambda x: 0 < abs(x) <= 1, actual_value)))

    assert all_metrics_correct


def test_cv_ts_and_cluster_raise():
    task = Task(task_type=TaskTypesEnum.ts_forecasting)
    dataset_to_compose, dataset_to_validate = get_data(task)
    models_repo = OperationTypesRepository()
    available_model_types, _ = models_repo.suitable_operation(task_type=task.task_type, tags=['simple'])
    composer_requirements = GPComposerRequirements(primary=available_model_types,
                                                   secondary=available_model_types,
                                                   cv_folds=4)
    metric_function = ClassificationMetricsEnum.ROCAUC_penalty
    builder = GPComposerBuilder(task).with_requirements(composer_requirements).with_metrics(metric_function)
    composer = builder.build()

    with pytest.raises(NotImplementedError):
        composer.compose_chain(data=dataset_to_compose, is_visualise=False)

    task = Task(task_type=TaskTypesEnum.clustering)
    dataset_to_compose, dataset_to_validate = get_data(task)
    metric_function = ClusteringMetricsEnum.silhouette
    builder = GPComposerBuilder(task).with_requirements(composer_requirements).with_metrics(metric_function)
    composer = builder.build()

    with pytest.raises(NotImplementedError):
        composer.compose_chain(data=dataset_to_compose, is_visualise=False)


def test_cv_min_kfolds_raise():
    task = Task(task_type=TaskTypesEnum.classification)
    models_repo = OperationTypesRepository()
    available_model_types, _ = models_repo.suitable_operation(task_type=task.task_type, tags=['simple'])

    with pytest.raises(ValueError):
        GPComposerRequirements(primary=available_model_types, secondary=available_model_types, cv_folds=2)


def test_composer_with_cv_optimization_correct():
    task = Task(task_type=TaskTypesEnum.classification)
    dataset_to_compose, dataset_to_validate = get_data(task)

    models_repo = OperationTypesRepository()
    available_model_types, _ = models_repo.suitable_operation(task_type=task.task_type, tags=['simple'])

    metric_function = [ClassificationMetricsEnum.ROCAUC_penalty,
                       ClassificationMetricsEnum.accuracy,
                       ClassificationMetricsEnum.logloss]

    composer_requirements = GPComposerRequirements(primary=available_model_types,
                                                   secondary=available_model_types,
                                                   max_lead_time=timedelta(minutes=1),
                                                   num_of_generations=3, cv_folds=3)

    builder = GPComposerBuilder(task).with_requirements(composer_requirements).with_metrics(metric_function)
    composer = builder.build()

    chain_evo_composed = composer.compose_chain(data=dataset_to_compose, is_visualise=False)[0]

    assert isinstance(chain_evo_composed, Chain)

    chain_evo_composed.fit(input_data=dataset_to_compose)
    predicted = chain_evo_composed.predict(dataset_to_validate)
    roc_on_valid_evo_composed = roc_auc(y_score=predicted.predict, y_true=dataset_to_validate.target)

    assert roc_on_valid_evo_composed > 0


def test_cv_api_correct():
    composer_params = {'max_depth': 1,
                       'max_arity': 2,
                       'learning_time': 0.0001,
                       'preset': 'ultra_light',
                       'cv_folds': 10}
    task = Task(task_type=TaskTypesEnum.classification)
    dataset_to_compose, dataset_to_validate = get_data(task)
    model = Fedot(problem='classification', composer_params=composer_params, verbose_level=2)
    fedot_model = model.fit(features=dataset_to_compose)
    prediction = model.predict(features=dataset_to_validate)
    metric = model.get_metrics()

    assert isinstance(fedot_model, Chain)
    assert len(prediction) == len(dataset_to_validate.target)
    assert metric['f1'] > 0
