import pandas as pd
import xgboost as xgb

from core.models.data import InputData
from core.repository.tasks import Task, TaskTypesEnum


def run_xgboost(train_file_path: str, test_file_path: str, target_name: str, task: TaskTypesEnum):
    train_datafile = pd.read_csv(train_file_path)
    test_datafile = pd.read_csv(test_file_path)
    if task is TaskTypesEnum.classification:
        train_data = InputData.from_csv(train_file_path)
        test_data = InputData.from_csv(test_file_path)
        true_target = test_datafile[target_name]
        model = xgb.XGBClassifier(max_depth=2, learning_rate=1.0, objective='binary:logistic')
        model.fit(train_data.features, train_data.target)
        predicted = model.predict_proba(test_data.features)[:, 1]
    elif task is TaskTypesEnum.regression:
        xgbr = xgb.XGBRegressor(max_depth=3, learning_rate=0.3, n_estimators=300, objective='reg:squarederror')
        x_train, y_train = train_datafile.iloc[:, :-1], train_datafile.iloc[:, -1]
        x_test, true_target = test_datafile.iloc[:, :-1], test_datafile.iloc[:, -1]
        xgbr.fit(x_train, y_train)
        predicted = xgbr.predict(x_test)
    else:
        raise NotImplementedError()
    return true_target, predicted


def run_xgb_classifier(train_file: str, test_file: str):
    train_data = InputData.from_csv(train_file)
    test_data = InputData.from_csv(test_file)

    model = xgb.XGBClassifier()
    model.fit(train_data.features, train_data.target)

    predicted = model.predict_proba(test_data.features)[:, 1]
    true_target = test_data.target

    return true_target, predicted
