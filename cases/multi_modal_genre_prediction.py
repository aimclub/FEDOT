
import datetime

from sklearn.metrics import f1_score

from examples.advanced.multi_modal_pipeline import calculate_validation_metric, \
    generate_initial_pipeline_and_data, prepare_multi_modal_data
from fedot.core.caching.pipelines_cache import OperationsCache

from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters, GeneticSchemeTypesEnum
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def run_multi_modal_case(files_path, is_visualise=True, timeout=datetime.timedelta(minutes=1)):
    task = Task(TaskTypesEnum.classification)
    images_size = (224, 224)

    data = prepare_multi_modal_data(files_path, task, images_size)

    initial_pipeline, fit_data, predict_data = generate_initial_pipeline_and_data(data, with_split=True)

    # the search of the models provided by the framework that can be used as nodes in a pipeline for the selected task
    available_model_types = get_operations_for_task(task=task, mode='model')

    # the choice of the metric for the pipeline quality assessment during composition
    metric_function = ClassificationMetricsEnum.f1
    # the choice and initialisation of the GP search
    composer_requirements = PipelineComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=3,
        max_depth=5, pop_size=5, num_of_generations=5,
        crossover_prob=0.8, mutation_prob=0.8, timeout=timeout)

    # GP optimiser parameters choice
    scheme_type = GeneticSchemeTypesEnum.parameter_free
    optimiser_parameters = GPGraphOptimiserParameters(genetic_scheme_type=scheme_type)

    # Create builder for composer and set composer params
    # the multi modal template (with data sources) is passed as initial assumption for composer
    builder = ComposerBuilder(task=task) \
        .with_requirements(composer_requirements) \
        .with_metrics(metric_function) \
        .with_optimiser_params(parameters=optimiser_parameters) \
        .with_initial_pipelines([initial_pipeline]) \
        .with_cache(OperationsCache())

    # Create GP-based composer
    composer = builder.build()

    # the optimal pipeline generation by composition - the most time-consuming task
    pipeline_evo_composed = composer.compose_pipeline(data=fit_data)
    pipeline_evo_composed.print_structure()

    # tuning of the composed pipeline
    pipeline_tuner = PipelineTuner(pipeline=pipeline_evo_composed, task=task, iterations=15)
    tuned_pipeline = pipeline_tuner.tune_pipeline(input_data=fit_data,
                                                  loss_function=f1_score,
                                                  loss_params={'average': 'micro'})
    tuned_pipeline.print_structure()
    tuned_pipeline.fit(input_data=fit_data)

    if is_visualise:
        tuned_pipeline.show()

    prediction = tuned_pipeline.predict(predict_data, output_mode='labels')
    err = calculate_validation_metric(predict_data, prediction)

    print(f'F1 micro for validation sample is {err}')
    return err


def download_mmdb_dataset():
    # TODO change to uploadable full dataset
    pass


if __name__ == '__main__':
    download_mmdb_dataset()

    run_multi_modal_case('cases/data/mm_imdb', is_visualise=True)
