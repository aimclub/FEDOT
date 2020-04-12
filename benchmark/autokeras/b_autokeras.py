import autokeras as ak
from sklearn.metrics import mean_squared_error as mse

from benchmark.benchmark_utils import get_scoring_case_data_paths
from core.models.data import InputData
from sklearn.metrics import roc_auc_score

MAX_TRIAL = 20
EPOCH = 1000


def run_autokeras(train_file_path: str, test_file_path: str, case_name: str = 'default', is_classification=True):
    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)

    # TODO Save model to file
    # task = 'clss' if is_classification else 'reg'
    # filename = f"{case_name}_m{MAX_TRIAL}_{task}"

    estimator = ak.StructuredDataClassifier if is_classification else ak.StructuredDataRegressor
    model = estimator(max_trials=MAX_TRIAL)

    model.fit(train_data.features, train_data.target, epochs=EPOCH)

    predicted = model.predict(test_data.features)

    if is_classification:
        result_metric = f'Autokeras roc_auc: {roc_auc_score(test_data.target, predicted)}'
    else:
        result_metric = f'MSE: {mse(test_data.target, predicted)}'

    return result_metric


if __name__ == '__main__':
    train_file, test_file = get_scoring_case_data_paths()
    run_autokeras(train_file, test_file)
