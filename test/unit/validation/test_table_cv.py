import os
from datetime import timedelta

import pytest
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.api.main import Fedot
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.log import default_log
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.validation.compose.tabular import table_metric_calculation
from fedot.core.validation.tune.tabular import cv_tabular_predictions
from test.unit.api.test_api_cli_params import project_root_path
from test.unit.models.test_model import classification_dataset
from test.unit.tasks.test_classification import get_iris_data, pipeline_simple

_ = classification_dataset


def sample_pipeline():
    return Pipeline(SecondaryNode(operation_type='logit',
                                  nodes_from=[PrimaryNode(operation_type='rf'),
                                              PrimaryNode(operation_type='scaling')]))


def get_data(task):
    file_path = os.path.join(project_root_path, 'test/data/simple_classification.csv')
    input_data = InputData.from_csv(file_path, task=task)
    dataset_to_compose, dataset_to_validate = train_test_data_setup(input_data)

    return dataset_to_compose, dataset_to_validate


def test_cv_multiple_metrics_evaluated_correct(classification_dataset):
    pipeline = sample_pipeline()
    log = default_log(__name__)

    actual_value = table_metric_calculation(pipeline=pipeline, reference_data=classification_dataset,
                                            cv_folds=3,
                                            metrics=[ClassificationMetricsEnum.ROCAUC_penalty,
                                                     ClassificationMetricsEnum.accuracy,
                                                     ClassificationMetricsEnum.logloss],
                                            log=log)
    all_metrics_correct = all(list(map(lambda x: 0 < abs(x) <= 1, actual_value)))

    assert all_metrics_correct


def test_cv_min_kfolds_raise():
    task = Task(task_type=TaskTypesEnum.classification)
    models_repo = OperationTypesRepository()
    available_model_types, _ = models_repo.suitable_operation(task_type=task.task_type, tags=['simple'])

    with pytest.raises(ValueError):
        PipelineComposerRequirements(primary=available_model_types, secondary=available_model_types, cv_folds=1)


def test_tuner_cv_classification_correct():
    folds = 2
    dataset = get_iris_data()

    simple_pipeline = pipeline_simple()
    tuned = simple_pipeline.fine_tune_all_nodes(loss_function=roc_auc,
                                                loss_params={"multi_class": "ovr"},
                                                input_data=dataset,
                                                iterations=1, timeout=1,
                                                cv_folds=folds)
    assert tuned


def test_cv_tabular_predictions_correct():
    folds = 2
    dataset = get_iris_data()

    simple_pipeline = pipeline_simple()
    predictions, target = cv_tabular_predictions(pipeline=simple_pipeline,
                                                 reference_data=dataset,
                                                 cv_folds=folds)
    dataset_size = len(dataset.features)
    predictions_size = len(predictions)
    target_size = len(target)
    assert dataset_size == predictions_size
    assert dataset_size == target_size


def test_composer_with_cv_optimization_correct():
    task = Task(task_type=TaskTypesEnum.classification)
    dataset_to_compose, dataset_to_validate = get_data(task)

    models_repo = OperationTypesRepository()
    available_model_types, _ = models_repo.suitable_operation(task_type=task.task_type, tags=['simple'])

    metric_function = [ClassificationMetricsEnum.ROCAUC_penalty,
                       ClassificationMetricsEnum.accuracy,
                       ClassificationMetricsEnum.logloss]

    composer_requirements = PipelineComposerRequirements(primary=available_model_types,
                                                         secondary=available_model_types,
                                                         timeout=timedelta(minutes=0.2),
                                                         num_of_generations=2, cv_folds=3)

    builder = ComposerBuilder(task).with_requirements(composer_requirements).with_metrics(metric_function)
    composer = builder.build()

    pipeline_evo_composed = composer.compose_pipeline(data=dataset_to_compose, is_visualise=False)[0]

    assert isinstance(pipeline_evo_composed, Pipeline)

    pipeline_evo_composed.fit(input_data=dataset_to_compose)
    predicted = pipeline_evo_composed.predict(dataset_to_validate)
    roc_on_valid_evo_composed = roc_auc(y_score=predicted.predict, y_true=dataset_to_validate.target)

    assert roc_on_valid_evo_composed > 0


def test_cv_api_correct():
    composer_params = {'max_depth': 1,
                       'max_arity': 2,
                       'timeout': 0.1,
                       'pop_size': 4,
                       'preset': 'fast_train',
                       'cv_folds': 2}
    task = Task(task_type=TaskTypesEnum.classification)
    dataset_to_compose, dataset_to_validate = get_data(task)
    model = Fedot(problem='classification', composer_params=composer_params, verbose_level=2)
    fedot_model = model.fit(features=dataset_to_compose)
    prediction = model.predict(features=dataset_to_validate)
    metric = model.get_metrics()

    assert isinstance(fedot_model, Pipeline)
    assert len(prediction) == len(dataset_to_validate.target)
    assert metric['f1'] > 0
