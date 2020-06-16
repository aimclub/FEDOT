import datetime
import random
from core.composer.gp_composer.gp_composer import GPComposer, GPComposerRequirements
from core.models.model import *
from core.repository.dataset_types import NumericalDataTypesEnum, CategoricalDataTypesEnum
from core.repository.model_types_repository import (
    ModelMetaInfoTemplate,
    ModelTypesRepository
)
from core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from core.repository.task_types import MachineLearningTasksEnum
from benchmark.benchmark_utils import get_models_hyperparameters
from examples.utils import get_multi_clf_data_paths, save_csv

random.seed(1)
np.random.seed(1)

file_path_first = r'./example1.xlsx'
file_path_second = r'./example2.xlsx'
name_of_dataset_second = 'example_2'
file_path_third = r'./example3.xlsx'
name_of_dataset_third = 'example_3'

train_file_path, test_file_path = get_multi_clf_data_paths(file_path_first)


def GetModel(train_file_path, cur_lead_time: int = 10, vis_flag: bool = False):
    problem_class = MachineLearningTasksEnum.classification
    dataset_to_compose = InputData.from_csv(train_file_path)
    models_hyperparameters = get_models_hyperparameters()['FEDOT']
    generations = models_hyperparameters['GENERATIONS']
    population_size = models_hyperparameters['POPULATION_SIZE']

    # the search of the models provided by the framework that can be used as nodes in a chain for the selected task
    models_repo = ModelTypesRepository()
    available_model_types, _ = models_repo.search_models(
        desired_metainfo=ModelMetaInfoTemplate(input_type=NumericalDataTypesEnum.table,
                                               output_type=CategoricalDataTypesEnum.vector,
                                               task_type=problem_class,
                                               can_be_initial=True,
                                               can_be_secondary=True))

    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=2,
        max_depth=3, pop_size=population_size, num_of_generations=generations,
        crossover_prob=0.8, mutation_prob=0.8, max_lead_time=datetime.timedelta(minutes=cur_lead_time))

    # Create GP-based composer
    composer = GPComposer()

    # the optimal chain generation by composition - the most time-consuming task
    chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                initial_chain=None,
                                                composer_requirements=composer_requirements,
                                                metrics=metric_function, is_visualise=vis_flag)
    chain_evo_composed.fit(input_data=dataset_to_compose, verbose=True)

    return chain_evo_composed


def ApplyModelToData(model, initial_file_path, name_of_dataset):
    df, test_file_path = save_csv(initial_file_path, name_of_dataset)
    dataset_to_validate = InputData.from_csv(test_file_path, target_flag=True)
    evo_predicted = model.predict(dataset_to_validate)
    df['forecast'] = evo_predicted.predict.tolist()
    return df


model = GetModel(train_file_path)
result_first = ApplyModelToData(model, file_path_second, name_of_dataset_second)
result_second = ApplyModelToData(model, file_path_third, name_of_dataset_third)

print(result_first)
print(result_second)
