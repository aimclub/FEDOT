import os
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from matplotlib import pyplot as plt
from xgboost import XGBClassifier, XGBRegressor

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.utils import default_fedot_data_dir


class FedotXGBoostImplementation(ModelImplementation):
    __operation_params = ['n_jobs', 'use_eval_set']

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

        self.model_params = {k: v for k, v in self.params.to_dict().items() if k not in self.__operation_params}
        self.model = None

    def fit(self, input_data: InputData):
        input_data = input_data.get_not_encoded_data()

        if self.params.get('use_eval_set'):
            train_input, eval_input = train_test_data_setup(input_data)

            train_input = self.convert_to_dataframe(train_input)
            eval_input = self.convert_to_dataframe(eval_input)

            train_x, train_y = train_input.drop(columns=['target']), train_input['target']
            eval_x, eval_y = eval_input.drop(columns=['target']), eval_input['target']

            # TODO: Get metric for evaluating on validation
            if self.classes_ is None:
                eval_metric = 'rmse'
            elif len(self.classes_) < 3:
                eval_metric = 'auc'
            else:
                eval_metric = 'mlogloss'

            self.model.fit(X=train_x, y=train_y, eval_set=[(eval_x, eval_y)], eval_metric=eval_metric)

        else:
            train_data = self.convert_to_dataframe(input_data)
            train_x, train_y = train_data.drop(columns=['target']), train_data['target']
            self.model.fit(X=train_x, y=train_y)

        return self.model

    def predict(self, input_data: InputData):
        input_data = self.convert_to_dataframe(input_data.get_not_encoded_data())
        train_x, _ = input_data.drop(columns=['target']), input_data['target']
        prediction = self.model.predict(train_x)

        return prediction

    def get_feature_importance(self) -> list:
        return self.model.features_importances_

    @staticmethod
    def convert_to_dataframe(data: Optional[InputData]):
        dataframe = pd.DataFrame(data=data.features, columns=data.features_names)
        dataframe['target'] = data.target

        if data.categorical_idx is not None:
            for col in dataframe.columns[data.categorical_idx]:
                dataframe[col] = dataframe[col].astype('category')

        if data.numerical_idx is not None:
            for col in dataframe.columns[data.numerical_idx]:
                dataframe[col] = dataframe[col].astype('float')

        return dataframe

    def plot_feature_importance(self):
        plot_feature_importance(
            self.model.feature_names_, self.model.get_boosters().features_importances_
        )


class FedotXGBoostClassificationImplementation(FedotXGBoostImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.classes_ = None
        self.model = XGBClassifier(**self.model_params)

    def fit(self, input_data: InputData):
        self.classes_ = np.unique(np.array(input_data.target))
        return super().fit(input_data=input_data)

    def predict_proba(self, input_data: InputData):
        input_data = self.convert_to_dataframe(input_data.get_not_encoded_data())
        train_x, _ = input_data.drop(columns=['target']), input_data['target']
        prediction = self.model.predict_proba(train_x)
        return prediction


class FedotXGBoostRegressionImplementation(FedotXGBoostImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.classes_ = None
        self.model = XGBRegressor(**self.model_params)


class FedotCatBoostImplementation(ModelImplementation):
    __operation_params = ['use_eval_set', 'n_jobs']

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

        self.check_and_update_params()

        self.model_params = {k: v for k, v in self.params.to_dict().items() if k not in self.__operation_params}
        self.model = None

    def fit(self, input_data: InputData):
        input_data = input_data.get_not_encoded_data()

        if self.params.get('use_eval_set'):
            # TODO: Using this method for tuning
            train_input, eval_input = train_test_data_setup(input_data)

            train_input = self.convert_to_pool(train_input)
            eval_input = self.convert_to_pool(eval_input)

            self.model.fit(X=train_input, eval_set=eval_input)

        else:
            train_input = self.convert_to_pool(input_data)

            self.model.fit(train_input)

        return self.model

    def predict(self, input_data: InputData):
        prediction = self.model.predict(input_data.get_not_encoded_data().features)

        return prediction

    def check_and_update_params(self):
        n_jobs = self.params.get('n_jobs')
        self.params.update(thread_count=n_jobs)

        use_best_model = self.params.get('use_best_model')
        early_stopping_rounds = self.params.get('early_stopping_rounds')
        use_eval_set = self.params.get('use_eval_set')

        if (use_best_model or isinstance(early_stopping_rounds, int)) and not use_eval_set:
            self.params.update(use_best_model=False, early_stopping_rounds=False)

    @staticmethod
    def convert_to_pool(data: Optional[InputData]):
        return Pool(
            data=data.features,
            label=data.target,
            cat_features=data.categorical_idx,
            feature_names=data.features_names.tolist() if data.features_names is not None else None
        )

    def save_model(self, model_name: str = 'catboost'):
        save_path = os.path.join(default_fedot_data_dir(), f'catboost/{model_name}.cbm')
        self.model.save_model(save_path, format='cbm')

    def load_model(self, path):
        self.model = CatBoostClassifier()
        self.model.load_model(path)

    def get_feature_importance(self) -> (list, list):
        """ Return feature importance -> (feature_id (string), feature_importance (float)) """
        return self.model.get_feature_importance(prettified=True)

    def plot_feature_importance(self):
        plot_feature_importance(self.model.feature_names_, self.model.features_importances_)


class FedotCatBoostClassificationImplementation(FedotCatBoostImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = CatBoostClassifier(**self.model_params)
        self.classes_ = None

    def fit(self, input_data: InputData):
        self.classes_ = np.unique(np.array(input_data.target))
        return super().fit(input_data=input_data)

    def predict_proba(self, input_data: InputData):
        prediction = self.model.predict_proba(input_data.get_not_encoded_data().features)
        return prediction


class FedotCatBoostRegressionImplementation(FedotCatBoostImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = CatBoostRegressor(**self.model_params)


def plot_feature_importance(feature_names, feature_importance):
    fi = pd.DataFrame(index=feature_names)
    fi['importance'] = feature_importance

    fi.loc[fi['importance'] > 0.1].sort_values('importance').plot(
        kind='barh', figsize=(16, 9), title='Feature Importance')

    plt.show()
