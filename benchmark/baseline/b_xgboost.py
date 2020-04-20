import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from benchmark.benchmark_utils import get_scoring_case_data_paths
from core.models.data import InputData


def run_xgboost(train_file_path: str, test_file_path: str, target_name: str):
    train_dataframe = pd.read_csv(train_file_path)
    test_dataframe = pd.read_csv(test_file_path)

    train_data = xgb.DMatrix(train_dataframe, label=train_dataframe[target_name])
    test_data = xgb.DMatrix(test_dataframe, label=test_dataframe[target_name])

    xgb_params = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
    num_round = 20

    trained_model = xgb.train(params=xgb_params, dtrain=train_data, num_boost_round=num_round)

    predicted = trained_model.predict(test_data)

    roc_auc_value = round(roc_auc_score(test_data.get_label(), predicted), 3)

    return roc_auc_value


def run_xgb_classifier(train_file: str, test_file: str):
    train_data = InputData.from_csv(train_file)
    test_data = InputData.from_csv(test_file)

    model = XGBClassifier()
    model.fit(train_data.features, train_data.target)

    predicted = model.predict_proba(test_data.features)[:, 1]

    roc_auc_value = round(roc_auc_score(test_data.target, predicted), 3)

    return roc_auc_value


if __name__ == '__main__':
    train_file, test_file = get_scoring_case_data_paths()

    xgb_booster_roc_auc = run_xgboost(train_file, test_file, target_name='default')
    xgb_cls_roc_auc = run_xgb_classifier(train_file, test_file)

    print(f'XGBoost_Booster roc_auc metric: {xgb_booster_roc_auc}')
    print(f'XGBClassifier roc_auc metric: {xgb_cls_roc_auc}')
