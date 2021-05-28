import datetime

from examples.multi_modal_chain import prepare_multi_modal_data, \
    generate_initial_chain_and_data, \
    calculate_validation_metric
from fedot.core.composer.gp_composer.gp_composer import GPComposerBuilder, GPComposerRequirements
from fedot.core.composer.optimisers.gp_comp.gp_optimiser import GPChainOptimiserParameters, GeneticSchemeTypesEnum
from fedot.core.log import default_log
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def run_multi_modal_case(files_path, is_visualise=False):
    task = Task(TaskTypesEnum.classification)
    images_size = (128, 128)

    train_num, test_num, train_img, test_img, train_text, test_text = prepare_multi_modal_data(files_path, task,
                                                                                               images_size)

    chain, fit_data, predict_data = generate_initial_chain_and_data(images_size,
                                                                    train_num, test_num,
                                                                    train_img, test_img,
                                                                    train_text, test_text)

    # the search of the models provided by the framework that can be used as nodes in a chain for the selected task
    available_model_types = get_operations_for_task(task=task, mode='models')

    # the choice of the metric for the chain quality assessment during composition
    metric_function = ClassificationMetricsEnum.ROCAUC_penalty
    # the choice and initialisation of the GP search
    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=3,
        max_depth=3, pop_size=5, num_of_generations=5,
        crossover_prob=0.8, mutation_prob=0.8, max_lead_time=datetime.timedelta(minutes=2))

    # GP optimiser parameters choice
    scheme_type = GeneticSchemeTypesEnum.parameter_free
    optimiser_parameters = GPChainOptimiserParameters(genetic_scheme_type=scheme_type)

    # Create builder for composer and set composer params
    logger = default_log('FEDOT logger', verbose_level=4)

    # the multi modal template (with data sources) is passed as inital assumption for composer
    builder = GPComposerBuilder(task=task).with_requirements(composer_requirements). \
        with_metrics(metric_function).with_optimiser_parameters(optimiser_parameters).with_logger(logger=logger). \
        with_initial_chain(chain).with_cache('multi_modal_opt.cache')

    # Create GP-based composer
    composer = builder.build()

    # the optimal chain generation by composition - the most time-consuming task
    chain_evo_composed = composer.compose_chain(data=fit_data,
                                                is_visualise=True)

    chain_evo_composed.fit(input_data=fit_data)

    if is_visualise:
        chain_evo_composed.show()

    prediction = chain_evo_composed.predict(predict_data)

    err = calculate_validation_metric(prediction, test_num)

    print(f'ROC AUC for validation sample is {err}')

    return err


def download_mmdb_dataset():
    # TODO change to uploadable full dataset
    pass


if __name__ == '__main__':
    download_mmdb_dataset()
    run_multi_modal_case('cases/data/mm_imdb', is_visualise=True)
