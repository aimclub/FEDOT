import datetime

from examples.advanced.multi_modal_pipeline import calculate_validation_metric, generate_initial_pipeline_and_data, \
    prepare_multi_modal_data
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.log import default_log
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters, GeneticSchemeTypesEnum
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def run_multi_modal_case(files_path, is_visualise=False, timeout=datetime.timedelta(minutes=2)):
    task = Task(TaskTypesEnum.classification)
    images_size = (128, 128)

    train_num, test_num, train_img, test_img, train_text, test_text = prepare_multi_modal_data(files_path, task,
                                                                                               images_size)

    pipeline, fit_data, predict_data = generate_initial_pipeline_and_data(images_size,
                                                                          train_num, test_num,
                                                                          train_img, test_img,
                                                                          train_text, test_text)

    # the search of the models provided by the framework that can be used as nodes in a pipeline for the selected task
    available_model_types = get_operations_for_task(task=task, mode='model')

    # the choice of the metric for the pipeline quality assessment during composition
    metric_function = ClassificationMetricsEnum.ROCAUC_penalty
    # the choice and initialisation of the GP search
    composer_requirements = PipelineComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=3,
        max_depth=3, pop_size=5, num_of_generations=5,
        crossover_prob=0.8, mutation_prob=0.8, timeout=timeout)

    # GP optimiser parameters choice
    scheme_type = GeneticSchemeTypesEnum.parameter_free
    optimiser_parameters = GPGraphOptimiserParameters(genetic_scheme_type=scheme_type)

    # Create builder for composer and set composer params
    logger = default_log('FEDOT logger', verbose_level=4)

    # the multi modal template (with data sources) is passed as inital assumption for composer
    builder = ComposerBuilder(task=task).with_requirements(composer_requirements). \
        with_metrics(metric_function).with_optimiser(parameters=optimiser_parameters).with_logger(logger=logger). \
        with_initial_pipelines([pipeline]).with_cache('multi_modal_opt.cache')

    # Create GP-based composer
    composer = builder.build()

    # the optimal pipeline generation by composition - the most time-consuming task
    pipeline_evo_composed = composer.compose_pipeline(data=fit_data,
                                                      is_visualise=True)

    pipeline_evo_composed.fit(input_data=fit_data)

    if is_visualise:
        pipeline_evo_composed.show()

    prediction = pipeline_evo_composed.predict(predict_data)

    err = calculate_validation_metric(prediction, test_num)

    print(f'ROC AUC for validation sample is {err}')

    return err


def download_mmdb_dataset():
    # TODO change to uploadable full dataset
    pass


if __name__ == '__main__':
    download_mmdb_dataset()
    run_multi_modal_case('cases/data/mm_imdb', is_visualise=True)
