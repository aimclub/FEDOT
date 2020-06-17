import datetime
import random
from core.composer.gp_composer.gp_composer import GPComposer, GPComposerRequirements
from core.models.model import *
from core.composer.chain import Chain
from core.repository.dataset_types import NumericalDataTypesEnum, CategoricalDataTypesEnum
from core.repository.model_types_repository import (
    ModelMetaInfoTemplate,
    ModelTypesRepository
)
from core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from core.repository.task_types import MachineLearningTasksEnum
from examples.utils import create_multi_clf_data_examples
from sklearn.metrics import roc_auc_score as roc_auc

from core.models.data import InputData

random.seed(1)
np.random.seed(1)


def get_model(train_file_path: str, cur_lead_time: int = 10):
    problem_class = MachineLearningTasksEnum.classification
    dataset_to_compose = InputData.from_csv(train_file_path)

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
        max_depth=3, pop_size=10, num_of_generations=10,
        crossover_prob=0.8, mutation_prob=0.8, max_lead_time=datetime.timedelta(minutes=cur_lead_time))

    # Create GP-based composer
    composer = GPComposer()

    # the optimal chain generation by composition - the most time-consuming task
    chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                initial_chain=None,
                                                composer_requirements=composer_requirements,
                                                metrics=metric_function, is_visualise=False)
    chain_evo_composed.fit(input_data=dataset_to_compose, verbose=True)

    return chain_evo_composed


def apply_model_to_data(model: Chain, initial_file_path: str, test_dataset_exist: bool = True):
    if test_dataset_exist:
        dataset_to_validate = InputData.from_csv(initial_file_path)
        evo_predicted = model.predict(dataset_to_validate)
        return evo_predicted.predict
    else:
        df, test_file_path = create_multi_clf_data_examples(initial_file_path, return_df=True)
        dataset_to_validate = InputData.from_csv(test_file_path, with_target=False)
        evo_predicted = model.predict(dataset_to_validate)
        df['forecast'] = evo_predicted.predict.tolist()
        return df


if __name__ == '__main__':
    file_path_first = r'./data/example1.xlsx'
    file_path_second = r'./data/example2.xlsx'

    train_file_path, test_file_path = create_multi_clf_data_examples(file_path_first)
    test_data = InputData.from_csv(test_file_path)

    fitted_model = get_model(train_file_path)
    test_prediction = apply_model_to_data(fitted_model, test_file_path)
    final_prediction = apply_model_to_data(fitted_model, file_path_second, test_dataset_exist=False)
    roc_auc_on_test = round(roc_auc(y_true=test_data.target,
                                    y_score=test_prediction,
                                    multi_class='ovo',
                                    average='macro'), 3)
    print(roc_auc_on_test)
