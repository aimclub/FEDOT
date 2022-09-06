import logging
import os
from datetime import timedelta
from functools import partial

import pytest
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.model_selection import KFold, StratifiedKFold

from fedot.api.main import Fedot
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.metrics import ROCAUC
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.objective import PipelineObjectiveEvaluate
from fedot.core.optimisers.objective.data_objective_advisor import DataObjectiveAdvisor
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.validation.split import tabular_cv_generator
from test.unit.api.test_api_cli_params import project_root_path
from test.unit.models.test_model import classification_dataset
from test.unit.tasks.test_classification import get_iris_data, pipeline_simple

_ = classification_dataset


def sample_pipeline():
    return Pipeline(SecondaryNode(operation_type='logit',
                                  nodes_from=[PrimaryNode(operation_type='rf'),
                                              PrimaryNode(operation_type='scaling')]))


def get_classification_data():
    file_path = os.path.join(project_root_path, 'test/data/simple_classification.csv')
    input_data = InputData.from_csv(file_path, task=Task(TaskTypesEnum.classification))
    return input_data


def test_cv_multiple_metrics_evaluated_correct(classification_dataset):
    pipeline = sample_pipeline()

    cv_folds = partial(tabular_cv_generator, classification_dataset, folds=3)
    metrics = [ClassificationMetricsEnum.ROCAUC_penalty,
               ClassificationMetricsEnum.accuracy,
               ClassificationMetricsEnum.logloss]
    objective_eval = PipelineObjectiveEvaluate(Objective(metrics), cv_folds)
    actual_values = objective_eval(pipeline).values
    all_metrics_correct = all(0 < abs(x) <= 1 for x in actual_values)

    assert all_metrics_correct


def test_kfold_advisor_works_correct_in_balanced_case():
    data = get_classification_data()
    advisor = DataObjectiveAdvisor()
    split_type = advisor.propose_kfold(data)
    assert split_type == KFold


def test_kfold_advisor_works_correct_in_imbalanced_case():
    data = get_classification_data()
    data.target[:-int(len(data.target) * 0.1)] = 0
    advisor = DataObjectiveAdvisor()
    split_type = advisor.propose_kfold(data)
    assert split_type == StratifiedKFold


def test_cv_min_kfolds_raise():
    task = Task(task_type=TaskTypesEnum.classification)
    models_repo = OperationTypesRepository()
    available_model_types = models_repo.suitable_operation(task_type=task.task_type, tags=['simple'])

    with pytest.raises(ValueError):
        PipelineComposerRequirements(primary=available_model_types, secondary=available_model_types, cv_folds=1)


def test_tuner_cv_classification_correct():
    folds = 5
    dataset = get_iris_data()

    simple_pipeline = pipeline_simple()
    tuned = simple_pipeline.fine_tune_all_nodes(loss_function=ROCAUC.metric,
                                                input_data=dataset,
                                                iterations=1, timeout=1,
                                                cv_folds=folds)
    assert tuned


def test_composer_with_cv_optimization_correct():
    task = Task(task_type=TaskTypesEnum.classification)
    dataset_to_compose, dataset_to_validate = train_test_data_setup(get_classification_data())

    models_repo = OperationTypesRepository()
    available_model_types = models_repo.suitable_operation(task_type=task.task_type, tags=['simple'])

    metric_function = [ClassificationMetricsEnum.ROCAUC_penalty,
                       ClassificationMetricsEnum.accuracy,
                       ClassificationMetricsEnum.logloss]

    composer_requirements = PipelineComposerRequirements(primary=available_model_types,
                                                         secondary=available_model_types,
                                                         timeout=timedelta(minutes=0.2),
                                                         num_of_generations=2, cv_folds=3,
                                                         num_of_generations=2, cv_folds=5,
                                                         logging_level_opt=logging.CRITICAL+1,
                                                         show_progress=False)

    builder = ComposerBuilder(task).with_requirements(composer_requirements).with_metrics(metric_function)
    composer = builder.build()

    pipeline_evo_composed = composer.compose_pipeline(data=dataset_to_compose)[0]

    assert isinstance(pipeline_evo_composed, Pipeline)

    pipeline_evo_composed.fit(input_data=dataset_to_compose)
    predicted = pipeline_evo_composed.predict(dataset_to_validate)
    roc_on_valid_evo_composed = roc_auc(y_score=predicted.predict, y_true=dataset_to_validate.target)

    assert roc_on_valid_evo_composed > 0


def test_cv_api_correct():
    composer_params = {'max_depth': 1,
                       'max_arity': 2,
                       'pop_size': 3,
                       'num_of_generations': 1,
                       'preset': 'fast_train',
                       'cv_folds': 2,
                       'show_progress': False}
    dataset_to_compose, dataset_to_validate = train_test_data_setup(get_classification_data())
    model = Fedot(problem='classification', logging_level=logging.INFO, **composer_params)
    fedot_model = model.fit(features=dataset_to_compose)
    prediction = model.predict(features=dataset_to_validate)
    metric = model.get_metrics()

    assert isinstance(fedot_model, Pipeline)
    assert len(prediction) == len(dataset_to_validate.target)
    assert metric['f1'] > 0
