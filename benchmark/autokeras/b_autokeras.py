import autokeras as ak

from benchmark.benchmark_utils import get_models_hyperparameters, get_scoring_case_data_paths
from core.models.data import InputData
from core.repository.tasks import Task, TaskTypesEnum


def run_autokeras(train_file_path: str, test_file_path: str, task: TaskTypesEnum,
                  case_name: str = 'default'):
    config_data = get_models_hyperparameters()['autokeras']
    max_trial = config_data['MAX_TRIAL']
    epoch = config_data['EPOCH']

    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)

    # TODO Save model to file

    if task is TaskTypesEnum.classification:
        estimator = ak.StructuredDataClassifier
    else:
        estimator = ak.StructuredDataRegressor

    model = estimator(max_trials=max_trial)

    model.fit(train_data.features, train_data.target, epochs=epoch)

    predicted = model.predict(test_data.features)

    return test_data.target, predicted


if __name__ == '__main__':
    train_file, test_file = get_scoring_case_data_paths()

    run_autokeras(train_file, test_file, task=TaskTypesEnum.classification)
