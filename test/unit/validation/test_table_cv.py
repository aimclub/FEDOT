import logging
from datetime import timedelta

import pytest

from fedot.api.api_utils.presets import PresetsEnum
from fedot.core.repository.operation_tags import ComplexityTagsEnum
from golem.core.tuning.simultaneous import SimultaneousTuner

from fedot import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.objective import PipelineObjectiveEvaluate
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.optimisers.objective.metrics_objective import MetricsObjective
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root
from test.integration.models.test_model import classification_dataset
from test.unit.tasks.test_classification import get_iris_data, pipeline_simple

_ = classification_dataset


def sample_pipeline():
    return Pipeline(PipelineNode(operation_type='logit',
                                 nodes_from=[PipelineNode(operation_type='rf'),
                                             PipelineNode(operation_type='scaling')]))


def get_classification_data():
    file_path = fedot_project_root().joinpath('test/data/simple_classification.csv')
    input_data = InputData.from_csv(file_path, task=Task(TaskTypesEnum.classification))
    return input_data


def test_cv_multiple_metrics_evaluated_correct(classification_dataset):
    pipeline = sample_pipeline()

    data_producer = DataSourceSplitter(cv_folds=5).build(classification_dataset)
    metrics = [ClassificationMetricsEnum.ROCAUC_penalty,
               ClassificationMetricsEnum.accuracy,
               ClassificationMetricsEnum.logloss]
    objective_eval = PipelineObjectiveEvaluate(MetricsObjective(metrics),
                                               data_producer=data_producer)
    actual_values = objective_eval(pipeline).values
    all_metrics_correct = all(0 < abs(x) <= 1 for x in actual_values)

    assert all_metrics_correct


def test_cv_min_kfolds_raise():
    task = Task(task_type=TaskTypesEnum.classification)
    models_repo = OperationTypesRepository()
    available_model_types = models_repo.suitable_operation(task_type=task.task_type, tags=[ComplexityTagsEnum.simple])

    with pytest.raises(ValueError):
        PipelineComposerRequirements(primary=available_model_types, secondary=available_model_types, cv_folds=1)


def test_tuner_cv_classification_correct():
    folds = 5
    dataset = get_iris_data()

    simple_pipeline = pipeline_simple()
    tuner = (
        TunerBuilder(dataset.task).with_tuner(SimultaneousTuner)
        .with_metric(ClassificationMetricsEnum.ROCAUC)
        .with_cv_folds(folds)
        .with_iterations(1)
        .with_timeout(timedelta(minutes=1))
        .build(dataset)
    )
    tuned = tuner.tune(simple_pipeline)
    assert tuned


def test_cv_api_correct():
    composer_params = {'max_depth': 1,
                       'max_arity': 2,
                       'pop_size': 3,
                       'num_of_generations': 1,
                       'preset': PresetsEnum.FAST_TRAIN,
                       'cv_folds': 2,
                       'show_progress': False,
                       'timeout': 0.3}
    dataset_to_compose, dataset_to_validate = train_test_data_setup(get_classification_data())
    model = Fedot(problem='classification', logging_level=logging.DEBUG, **composer_params)
    fedot_model = model.fit(features=dataset_to_compose)
    prediction = model.predict(features=dataset_to_validate)
    metric = model.get_metrics(metric_names='f1', rounding_order=1)

    assert isinstance(fedot_model, Pipeline)
    assert len(prediction) == len(dataset_to_validate.target)
    assert metric['f1'] > 0
