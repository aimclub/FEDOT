import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from core.repository.task_types import MachineLearningTasksEnum
from core.models.data import InputData


def run_xgboost(train_file_path: str, test_file_path: str, target_name: str, task: MachineLearningTasksEnum):
    train_datafile = pd.read_csv(train_file_path)
    test_datafile = pd.read_csv(test_file_path)
    if task is MachineLearningTasksEnum.classification:
        train_data = xgb.DMatrix(train_datafile, label=train_datafile[target_name])
        test_data = xgb.DMatrix(test_datafile, label=test_datafile[target_name])
        xgb_params = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
        num_round = 20
        trained_model = xgb.train(params=xgb_params, dtrain=train_data, num_boost_round=num_round)
        predicted = trained_model.predict(test_data)
        result_metric = round(roc_auc_score(test_data.get_label(), predicted), 3)
    elif task is MachineLearningTasksEnum.regression:
        xgbr = xgb.XGBRegressor(max_depth=3, learning_rate=0.3, n_estimators=300, objective='reg:squarederror')
        x_train, y_train = train_datafile.iloc[:, :-1], train_datafile.iloc[:, -1]
        x_test, y_test = test_datafile.iloc[:, :-1], test_datafile.iloc[:, -1]
        xgbr.fit(x_train, y_train)
        predicted = xgbr.predict(x_test)
        result_metric = round(mse(y_test, predicted), 3)
    return result_metric


def run_xgb_classifier(train_file: str, test_file: str):
    train_data = InputData.from_csv(train_file)
    test_data = InputData.from_csv(test_file)

    model = XGBClassifier()
    model.fit(train_data.features, train_data.target)

    predicted = model.predict_proba(test_data.features)[:, 1]

    roc_auc_value = round(roc_auc_score(test_data.target, predicted), 3)

    return roc_auc_value
