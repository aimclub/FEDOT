from datetime import timedelta

from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.repository.operation_tags import ModelTagsEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.unit.validation.test_table_cv import get_classification_data


def test_composer_with_cv_optimization_correct():
    task = Task(task_type=TaskTypesEnum.classification)
    dataset_to_compose, dataset_to_validate = train_test_data_setup(get_classification_data())

    models_repo = OperationTypesRepository()
    available_model_types = models_repo.suitable_operation(task_type=task.task_type, tags=[ModelTagsEnum.linear])

    metric_function = [ClassificationMetricsEnum.ROCAUC_penalty,
                       ClassificationMetricsEnum.accuracy,
                       ClassificationMetricsEnum.logloss]

    composer_requirements = PipelineComposerRequirements(primary=available_model_types,
                                                         secondary=available_model_types,
                                                         timeout=timedelta(minutes=0.2),
                                                         num_of_generations=2, cv_folds=5,
                                                         show_progress=False)

    builder = ComposerBuilder(task).with_requirements(composer_requirements).with_metrics(metric_function)
    composer = builder.build()

    pipeline_evo_composed = composer.compose_pipeline(data=dataset_to_compose)[0]

    assert isinstance(pipeline_evo_composed, Pipeline)

    pipeline_evo_composed.fit(input_data=dataset_to_compose)
    predicted = pipeline_evo_composed.predict(dataset_to_validate)
    roc_on_valid_evo_composed = roc_auc(y_score=predicted.predict, y_true=dataset_to_validate.target)

    assert roc_on_valid_evo_composed > 0
