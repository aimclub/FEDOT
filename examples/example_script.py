import datetime
import random
import pandas as pd
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

random.seed(1)
np.random.seed(1)

from benchmark.benchmark_utils import get_multi_clf_data_paths

train_file_path, test_file_path = get_multi_clf_data_paths()


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


def ApplyModelToData(model, test_file_path):
    """ Метод применения модели к данным.
    Parameters
    ----------
    model : обученная/настроенная модель.
    df : dataframe, к которому применяется модель
    """
    dataset_to_validate = InputData.from_csv(test_file_path)
    evo_predicted = model.predict(dataset_to_validate)
    # df['forecast'] = evo_predicted.predict(df)

    return evo_predicted.predict


# получаем откуда-то данные (от другой модели или из файла)
# df = pd.read_excel('example1.xlsx')

# обучаем модель на входных данных
model = GetModel(train_file_path)

# получаем откуда-то другие данные (от другой модели или из файла)
# df2 = pd.read_excel('example2.xlsx')

# применяем модель к новым данным
result = ApplyModelToData(model, test_file_path)
print(result)
# читаем еще один набор данных
# df3 = pd.read_excel('example3.xlsx')

# применяем модель еще раз к новым данным
# ApplyModelToData(model,df3)

# print(df3)
