from datetime import timedelta

from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.operations.cross_validation import cross_validation
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.composer.optimisers.gp_comp.gp_optimiser import GPChainOptimiserParameters
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.composer.gp_composer.gp_composer import GPComposerRequirements, GPComposerBuilder
from cases.credit_scoring_problem import get_scoring_data
from test.unit.models.test_model import classification_dataset


def sample_chain():
    return Chain(SecondaryNode(operation_type='logit',
                               nodes_from=[PrimaryNode(operation_type='xgboost'),
                                           PrimaryNode(operation_type='scaling')]))


def test_cv_metric_correct(classification_dataset):
    chain = sample_chain()

    actual_value = cross_validation(chain=chain, reference_data=classification_dataset, cv_folds=10,
                                    metrics=[ClassificationMetricsEnum.ROCAUC_penalty,
                                             ClassificationMetricsEnum.accuracy,
                                             ClassificationMetricsEnum.logloss])

    assert all(list(map(lambda x: x >= -1, actual_value)))


def test_cv_with_composer_optimisation_correct():
    full_path_train, full_path_test = get_scoring_data()
    task = Task(task_type=TaskTypesEnum.classification)
    dataset_to_compose = InputData.from_csv(full_path_train, task=task)
    dataset_to_validate = InputData.from_csv(full_path_test, task=task)

    models_repo = OperationTypesRepository()
    available_model_types, _ = models_repo.suitable_operation(task_type=task.task_type, tags=['simple'])

    metric_function = [ClassificationMetricsEnum.ROCAUC_penalty,
                       ClassificationMetricsEnum.accuracy,
                       ClassificationMetricsEnum.logloss]

    optimiser_parameters = GPChainOptimiserParameters(cv_folds=3)

    composer_requirements = GPComposerRequirements(primary=available_model_types,
                                                   secondary=available_model_types,
                                                   max_lead_time=timedelta(minutes=2),
                                                   num_of_generations=5)

    builder = GPComposerBuilder(task).with_requirements(composer_requirements)\
        .with_metrics(metric_function).with_optimiser_parameters(optimiser_parameters)
    composer = builder.build()

    chain_evo_composed = composer.compose_chain(data=dataset_to_compose, is_visualise=False)[0]

    assert isinstance(chain_evo_composed, Chain)

    chain_evo_composed.fit(input_data=dataset_to_compose)
    predicted = chain_evo_composed.predict(dataset_to_validate)
    roc_on_valid_evo_composed = roc_auc(y_score=predicted.predict, y_true=dataset_to_validate.target)

    assert roc_on_valid_evo_composed > 0
