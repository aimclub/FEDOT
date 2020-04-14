import autokeras as ak
from sklearn.metrics import mean_squared_error as mse

from benchmark.benchmark_utils import get_scoring_case_data_paths
from core.models.data import InputData
from sklearn.metrics import roc_auc_score


def run_autokeras(train_file_path: str, test_file_path: str, config_data: dict, case_name: str = 'default',
                  is_classification=True):
    max_trial = config_data['MAX_TRIAL']
    epoch = config_data['EPOCH']

    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)

    # TODO Save model to file
    # task = 'clss' if is_classification else 'reg'
    # filename = f"{case_name}_m{MAX_TRIAL}_{task}"

    estimator = ak.StructuredDataClassifier if is_classification else ak.StructuredDataRegressor
    model = estimator(max_trials=max_trial)

    model.fit(train_data.features, train_data.target, epochs=epoch)

    predicted = model.predict(test_data.features)

    if is_classification:
        result_metric = {'autokeras_roc_auc': round(roc_auc_score(test_data.target, predicted), 3)}
    else:
        result_metric = {'MSE': round(mse(test_data.target, predicted), 3)}

    return result_metric


if __name__ == '__main__':
    train_file, test_file = get_scoring_case_data_paths()

    autokeras_config = {'MAX_TRIAL': 20,
                        'EPOCH': 1000}

    run_autokeras(train_file, test_file, config_data=autokeras_config)
