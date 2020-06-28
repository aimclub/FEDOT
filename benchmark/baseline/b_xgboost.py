import xgboost as xgb

from core.models.data import InputData
from core.repository.tasks import TaskTypesEnum


def run_xgboost(params: 'ExecutionParams'):
    train_file_path = params.train_file
    test_file_path = params.test_file
    task = params.task

    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)

    if task is TaskTypesEnum.classification:
        model = xgb.XGBClassifier(max_depth=2, learning_rate=1.0, objective='binary:logistic')
        model.fit(train_data.features, train_data.target)
        predicted = model.predict_proba(test_data.features)[:, 1]
    elif task is TaskTypesEnum.regression:
        xgbr = xgb.XGBRegressor(max_depth=3, learning_rate=0.3, n_estimators=300,
                                objective='reg:squarederror')
        xgbr.fit(train_data.features, train_data.target)
        predicted = xgbr.predict(test_data.features)
    else:
        raise NotImplementedError()
    return test_data.target, predicted
