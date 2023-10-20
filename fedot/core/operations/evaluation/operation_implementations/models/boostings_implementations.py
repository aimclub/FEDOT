import os
from typing import Optional

import pandas as pd
from catboost import Pool, CatBoostClassifier, CatBoostRegressor
from matplotlib import pyplot as plt

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.utils import default_fedot_data_dir


class FedotCatBoostImplementation(ModelImplementation):
    __operation_params = ['use_eval_set', 'n_jobs']

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

        self.check_and_update_params()
        self.model_params = {k: v for k, v in self.params.to_dict().items() if k not in self.__operation_params}
        self.model = None

    def fit(self, input_data: InputData, eval_data: InputData = None):
        input_data = input_data.get_not_encoded_data()

        if eval_data is not None:
            eval_data = eval_data.get_not_encoded_data()

        if self.params.get('use_eval_set') or eval_data:
            train_input = self.convert_to_pool(input_data)
            eval_input = self.convert_to_pool(eval_data)

            self.model.fit(X=train_input, eval_set=eval_input)

        elif self.params.get('use_eval_set'):
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

        if use_best_model or early_stopping_rounds and not use_eval_set:
            self.params.update(use_best_model=False, early_stopping_rounds=False)

    @staticmethod
    def convert_to_pool(data: Optional[InputData]):
        return Pool(
            data=data.features,
            label=data.target,
            cat_features=data.categorical_idx,
            feature_names=data.features_names.tolist()
        )

    def save_model(self, model_name: str = 'catboost'):
        save_path = os.path.join(default_fedot_data_dir(), f'catboost/{model_name}.cbm')
        self.model.save_model(save_path, format='cbm')

        return save_path

    def load_model(self, path):
        self.model = CatBoostClassifier()
        self.model.load_model(path)


class FedotCatBoostClassificationImplementation(FedotCatBoostImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

        self.model = CatBoostClassifier(**self.model_params)

    def predict_proba(self, input_data: InputData):
        prediction = self.model.predict_proba(input_data.get_not_encoded_data().features)

        return prediction

    def get_feature_importance(self):
        return self.model.get_feature_importance(prettified=True)

    def plot_feature_importance(self):
        fi = pd.DataFrame(index=self.model.feature_names_)
        fi['importance'] = self.model.feature_importances_

        fi.loc[fi['importance'] > 0.1].sort_values('importance').plot(
            kind='barh', figsize=(16, 9), title='Feature Importance'
        )

        plt.show()


class FedotCatBoostRegressionImplementation(FedotCatBoostImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = CatBoostRegressor(**self.model_params)
