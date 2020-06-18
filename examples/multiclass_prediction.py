import datetime
import random

from sklearn.metrics import roc_auc_score as roc_auc

from core.composer.chain import Chain
from core.composer.gp_composer.gp_composer import GPComposer, GPComposerRequirements
from core.models.model import *
from core.repository.dataset_types import CategoricalDataTypesEnum, NumericalDataTypesEnum
from core.repository.model_types_repository import (
    ModelMetaInfoTemplate,
    ModelTypesRepository
)
from core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository
from core.repository.task_types import MachineLearningTasksEnum
from examples.utils import create_multi_clf_examples_from_excel
from core.utils import probs_to_labels

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
                                               can_be_secondary=True))

    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)
    available_model_types_for_multiclf = [x for x in available_model_types if x != ModelTypesIdsEnum.svc]

    composer_requirements = GPComposerRequirements(
        primary=available_model_types_for_multiclf, secondary=available_model_types,
        max_lead_time=datetime.timedelta(minutes=cur_lead_time))

    # Create GP-based composer
    composer = GPComposer()

    # the optimal chain generation by composition - the most time-consuming task
    chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                initial_chain=None,
                                                composer_requirements=composer_requirements,
                                                metrics=metric_function, is_visualise=False)
    chain_evo_composed.fit(input_data=dataset_to_compose)

    return chain_evo_composed


def apply_model_to_data(model: Chain, data_path: str):
    df, file_path = create_multi_clf_examples_from_excel(data_path, return_df=True)
    dataset_to_apply = InputData.from_csv(file_path, with_target=False)
    evo_predicted = model.predict(dataset_to_apply)
    df['forecast'] = probs_to_labels(evo_predicted.predict)
    return df


def validate_model_quality(model: Chain, data_path: str):
    dataset_to_validate = InputData.from_csv(data_path)
    predicted_labels = model.predict(dataset_to_validate).predict

    roc_auc_valid = round(roc_auc(y_true=test_data.target,
                                  y_score=predicted_labels,
                                  multi_class='ovo',
                                  average='macro'), 3)

    return roc_auc_valid


if __name__ == '__main__':
    file_path_first = r'./data/example1.xlsx'
    file_path_second = r'./data/example2.xlsx'
    file_path_third = r'./data/example3.xlsx'

    train_file_path, test_file_path = create_multi_clf_examples_from_excel(file_path_first)
    test_data = InputData.from_csv(test_file_path)

    fitted_model = get_model(train_file_path)

    roc_auc = validate_model_quality(fitted_model, test_file_path)
    print(roc_auc)

    final_prediction_first = apply_model_to_data(fitted_model, file_path_second)
    print(final_prediction_first['forecast'])

    final_prediction_second = apply_model_to_data(fitted_model, file_path_third)
    print(final_prediction_second['forecast'])
