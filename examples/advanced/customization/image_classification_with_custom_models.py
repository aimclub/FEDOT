import datetime
from pathlib import Path

import tensorflow as tf
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum
from golem.core.tuning.simultaneous import SimultaneousTuner
from hyperopt import hp
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.metrics_repository import ClassificationMetricsEnum, ComplexityMetricsEnum
from fedot.core.repository.operation_types_repository import get_operations_for_task, OperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import set_random_seed, fedot_project_root

custom_search_space = {'filter_1': {
    'r': {
        'hyperopt-dist': hp.uniformint,
        'sampling-scope': [-254, 254],
        'type': 'discrete'}
},
    'filter_2': {
        'r': {
            'hyperopt-dist': hp.uniformint,
            'sampling-scope': [-254, 254],
            'type': 'discrete'},
}
}


def calculate_validation_metric(predicted: OutputData, dataset_to_validate: InputData) -> float:
    # the quality assessment for the simulation results
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                            y_score=predicted.predict,
                            multi_class="ovo")
    return roc_auc_value


def cnn_composite_pipeline() -> Pipeline:
    node_first = PipelineNode('filter_1')

    node_second = PipelineNode('cnn_1', nodes_from=[node_first])

    node_final = PipelineNode('rf', nodes_from=[node_second])

    pipeline = Pipeline(node_final)
    return pipeline


def setup_repository():
    repo_folder = Path(fedot_project_root(), 'examples', 'advanced', 'customization',
                       'repositories')
    OperationTypesRepository.__repository_dict__ = {
        'model': {'file': Path(repo_folder, 'my_model_repository.json'), 'initialized_repo': None, 'default_tags': []},
        'data_operation': {'file': Path(repo_folder, 'my_data_operation_repository.json'),
                           'initialized_repo': None, 'default_tags': []}
    }

    OperationParameters.custom_default_params_path = Path(repo_folder,
                                                          'my_default_operation_params.json')


def run_image_classification_automl(train_dataset: tuple,
                                    test_dataset: tuple):
    task = Task(TaskTypesEnum.classification)

    setup_repository()

    x_train, y_train = train_dataset[0], train_dataset[1]
    x_test, y_test = test_dataset[0], test_dataset[1]

    dataset_to_train = InputData.from_image(images=x_train,
                                            labels=y_train,
                                            task=task)
    dataset_to_validate = InputData.from_image(images=x_test,
                                               labels=y_test,
                                               task=task)

    dataset_to_train = dataset_to_train.subset_range(0, min(100, dataset_to_train.features.shape[0]))

    initial_pipeline = cnn_composite_pipeline()
    initial_pipeline.show()
    initial_pipeline.fit(dataset_to_train)
    predictions = initial_pipeline.predict(dataset_to_validate)
    roc_auc_on_valid = calculate_validation_metric(predictions,
                                                   dataset_to_validate)

    print(roc_auc_on_valid)

    # the choice of the metric for the pipeline quality assessment during composition
    quality_metric = ClassificationMetricsEnum.f1
    complexity_metric = ComplexityMetricsEnum.node_number
    metrics = [quality_metric, complexity_metric]
    # the choice and initialisation of the GP search
    composer_requirements = PipelineComposerRequirements(
        primary=get_operations_for_task(task=task, mode='all'),
        timeout=datetime.timedelta(minutes=3),
        num_of_generations=20, n_jobs=1, cv_folds=None
    )

    pop_size = 5

    # search space for hyper-parametric mutation
    PipelineSearchSpace.pre_defined_custom_search_space = custom_search_space

    params = GPAlgorithmParameters(
        selection_types=[SelectionTypesEnum.spea2],
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
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

    replace_default_search_space = True
    cv_folds = 1
    search_space = PipelineSearchSpace(custom_search_space=custom_search_space,
                                       replace_default_search_space=replace_default_search_space)

    pipeline_evo_composed.predict(dataset_to_validate)

    # .with_cv_folds(cv_folds) \
    pipeline_tuner = TunerBuilder(dataset_to_train.task) \
        .with_tuner(SimultaneousTuner) \
        .with_metric(ClassificationMetricsEnum.ROCAUC) \
        .with_cv_folds(cv_folds) \
        .with_iterations(50) \
        .with_search_space(search_space).build(dataset_to_train)

    pipeline_tuner.tune(pipeline_evo_composed)

    predictions = pipeline_evo_composed.predict(dataset_to_validate)

    roc_auc_on_valid = calculate_validation_metric(predictions,
                                                   dataset_to_validate)
    return roc_auc_on_valid, dataset_to_train, dataset_to_validate


if __name__ == '__main__':
    set_random_seed(1)

    training_set, testing_set = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    roc_auc_on_valid, dataset_to_train, dataset_to_validate = run_image_classification_automl(
        train_dataset=training_set,
        test_dataset=testing_set)

    print(roc_auc_on_valid)
