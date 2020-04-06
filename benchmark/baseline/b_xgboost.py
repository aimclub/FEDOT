import xgboost as xgb
from xgboost import XGBClassifier

from benchmark.benchmark_utils import get_scoring_case_data_paths
from sklearn.metrics import roc_auc_score

from core.models.data import InputData


def run_xgboost(train_file_path: str, test_file_path: str):

    train_data = xgb.DMatrix(f'{train_file_path}?format=csv&label_column=11', silent=True)
    test_data = xgb.DMatrix(f'{test_file_path}?format=csv&label_column=11', silent=True)

    xgb_params = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
    num_round = 2

    trained_model = xgb.train(xgb_params, train_data, num_round)

    predicted = trained_model.predict(test_data)

    roc_auc_value = roc_auc_score(test_data.get_label(), predicted)

    print(f'XGBoost_Booster roc_auc metric: {roc_auc_value}')

    return roc_auc_value


def run_xgb_classifier(train_file: str, test_file: str):

    train_data = InputData.from_csv(train_file)
    test_data = InputData.from_csv(test_file)

    model = XGBClassifier()
    model.fit(train_data.features, train_data.target)

    predicted = model.predict(test_data.features)

    roc_auc_value = roc_auc_score(test_data.target, predicted)

    return roc_auc_value


if __name__ == '__main__':
    train_file, test_file = get_scoring_case_data_paths()

    xgb_booster_roc_auc = run_xgboost(train_file, test_file)
    xgb_cls_roc_auc = run_xgb_classifier(train_file, test_file)

    print(f'XGBoost_Booster roc_auc metric: {xgb_booster_roc_auc}')
    print(f'XGBClassifier roc_auc metric: {xgb_cls_roc_auc}')
