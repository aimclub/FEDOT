import autokeras as ak
from sklearn.metrics import mean_squared_error as mse

from benchmark.benchmark_model_types import ModelTypesEnum
from benchmark.benchmark_utils import get_scoring_case_data_paths, get_models_hyperparameters
from core.models.data import InputData
from sklearn.metrics import roc_auc_score

from core.repository.task_types import MachineLearningTasksEnum


def run_autokeras(train_file_path: str, test_file_path: str, config_data: dict, task: MachineLearningTasksEnum,
                  case_name: str = 'default'):
    max_trial = config_data['MAX_TRIAL']
    epoch = config_data['EPOCH']

    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)

    # TODO Save model to file

    if task is MachineLearningTasksEnum.classification:
        estimator = ak.StructuredDataClassifier
    else:
        estimator = ak.StructuredDataRegressor

    model = estimator(max_trials=max_trial)

    model.fit(train_data.features, train_data.target, epochs=epoch)

    predicted = model.predict(test_data.features)

    if task is MachineLearningTasksEnum.classification:
        result_metric = {'autokeras_roc_auc': round(roc_auc_score(test_data.target, predicted), 3)}
    else:
        result_metric = {'MSE': round(mse(test_data.target, predicted), 3)}

    return result_metric


if __name__ == '__main__':
    train_file, test_file = get_scoring_case_data_paths()

    autokeras_config = get_models_hyperparameters()['autokeras']

    run_autokeras(train_file, test_file, config_data=autokeras_config, task=MachineLearningTasksEnum.classification)
