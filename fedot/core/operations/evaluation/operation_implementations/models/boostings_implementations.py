from typing import Optional

import pandas as pd
from catboost import Pool, CatBoostClassifier, CatBoostRegressor
from matplotlib import pyplot as plt

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class FedotCatBoostImplementation(ModelImplementation):
    __operation_params = ['use_eval_set']

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

        default_params = {
            'iterations': 1000,
            'use_eval_set': False,
            'use_best_model': False,
            'early_stopping_rounds': 50
        }

        complete_params = {**default_params, **self.params.to_dict()}
        self.params.update(**complete_params)
        # TODO: Adding checking params compatibility with each other
        # self.check_params(self.params.to_dict())
        self.model_params = {k: v for k, v in self.params.to_dict().items() if k not in self.__operation_params}
        self.model = None

    def fit(self, input_data: InputData):
        if self.params.get('use_eval_set'):
            train_input, eval_input = train_test_data_setup(input_data)

            train_input = self.convert_to_pool(train_input)
            eval_input = self.convert_to_pool(eval_input)

            self.model.fit(
                X=train_input,
                eval_set=eval_input,
            )

        else:
            train_input = self.convert_to_pool(input_data)

            self.model.fit(train_input)

        return self.model

    def predict(self, input_data: InputData):
        prediction = self.model.predict(input_data.features)

        return prediction

    @staticmethod
    def check_params(params):
        if params['use_best_model'] or params['early_stopping_rounds'] and not params['use_eval_set']:
            params['use_best_model'] = False
            params['early_stopping_rounds'] = False

        return params

    @staticmethod
    def convert_to_pool(data: Optional[InputData]):
        return Pool(
            data=data.features,
            label=data.target,
            cat_features=data.cat_features,
            feature_names=data.features_names
        )


class FedotCatBoostClassificationImplementation(FedotCatBoostImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

        self.model = CatBoostClassifier(**self.model_params)

    def predict_proba(self, input_data: InputData):
        prediction = self.model.predict_proba(input_data.features)

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
