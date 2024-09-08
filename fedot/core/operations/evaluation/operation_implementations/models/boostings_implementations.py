import os
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from lightgbm import LGBMClassifier, LGBMRegressor
from lightgbm import early_stopping as lgbm_early_stopping
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

        self.check_and_update_params()

        self.model_params = {k: v for k, v in self.params.to_dict().items() if k not in self.__operation_params}
        self.model = None
        self.features_names = None

    def fit(self, input_data: InputData):
        self.features_names = input_data.features_names

        if self.params.get('enable_categorical'):
            input_data = input_data.get_not_encoded_data()

        if self.params.get('use_eval_set'):
            train_input, eval_input = train_test_data_setup(input_data)

            X_train, y_train = self.convert_to_dataframe(
                train_input, identify_cats=self.params.get('enable_categorical')
            )

            X_eval, y_eval = self.convert_to_dataframe(
                eval_input, identify_cats=self.params.get('enable_categorical')
            )

            self.model.eval_metric = self.set_eval_metric(self.classes_)

            self.model.fit(X=X_train, y=y_train, eval_set=[(X_eval, y_eval)], verbose=self.model_params['verbosity'])
        else:
            X_train, y_train = self.convert_to_dataframe(
                input_data, identify_cats=self.params.get('enable_categorical')
            )
            self.features_names = input_data.features_names

            self.model.fit(X=X_train, y=y_train, verbose=self.model_params['verbosity'])

        return self.model

    def predict(self, input_data: InputData):
        if self.params.get('enable_categorical'):
            input_data = input_data.get_not_encoded_data()

        X, _ = self.convert_to_dataframe(input_data, self.params.get('enable_categorical'))
        prediction = self.model.predict(X)

        return prediction

    def check_and_update_params(self):
        early_stopping_rounds = self.params.get('early_stopping_rounds')
        use_eval_set = self.params.get('use_eval_set')

        if isinstance(early_stopping_rounds, int) and not use_eval_set:
            self.params.update(early_stopping_rounds=False)

        booster = self.params.get('booster')
        enable_categorical = self.params.get('enable_categorical')

        if booster == 'gblinear' and enable_categorical:
            self.params.update(enable_categorical=False)

        if booster == 'gbtree' and enable_categorical:
            self.params.update(enable_categorical=False)

    def get_feature_importance(self) -> list:
        return self.model.features_importances_

    def plot_feature_importance(self, importance_type='weight'):
        model_output = self.model.get_booster().get_score()
        features_names = self.features_names
        plot_feature_importance(features_names, model_output.values())

    @staticmethod
    def convert_to_dataframe(data: Optional[InputData], identify_cats: bool):
        dataframe = pd.DataFrame(data=data.features)
        if data.target is not None and data.target.size > 0:
            dataframe['target'] = np.ravel(data.target)
        else:
            # TODO: temp workaround in case data.target is set to None intentionally
            #  for test.integration.models.test_model.check_predict_correct
            dataframe['target'] = np.zeros(len(data.features))

        if identify_cats and data.categorical_idx is not None:
            for col in dataframe.columns[data.categorical_idx]:
                dataframe[col] = dataframe[col].astype('category')

        if data.numerical_idx is not None:
            for col in dataframe.columns[data.numerical_idx]:
                dataframe[col] = dataframe[col].astype('float')

        return dataframe.drop(columns=['target']), dataframe['target']

    @staticmethod
    def set_eval_metric(n_classes):
        if n_classes is None:  # if n_classes is None -> regression
            eval_metric = 'rmse'
        elif len(n_classes) < 3:  # if n_classes < 3 -> bin class
            eval_metric = 'auc'
        else:  # else multiclass
            eval_metric = 'mlogloss'

        return eval_metric


class FedotXGBoostClassificationImplementation(FedotXGBoostImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.classes_ = None
        self.model = XGBClassifier(**self.model_params)

    def fit(self, input_data: InputData):
        self.classes_ = np.unique(np.array(input_data.target))
        return super().fit(input_data=input_data)

    def predict_proba(self, input_data: InputData):
        if self.params.get('enable_categorical'):
            input_data = input_data.get_not_encoded_data()

        X, _ = self.convert_to_dataframe(input_data, self.params.get('enable_categorical'))
        prediction = self.model.predict_proba(X)
        return prediction


class FedotXGBoostRegressionImplementation(FedotXGBoostImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.classes_ = None
        self.model = XGBRegressor(**self.model_params)


class FedotLightGBMImplementation(ModelImplementation):
    __operation_params = ['n_jobs', 'use_eval_set', 'enable_categorical']

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

        self.check_and_update_params()

        self.model_params = {k: v for k, v in self.params.to_dict().items() if k not in self.__operation_params}
        self.model = None
        self.features_names = None

    def fit(self, input_data: InputData):
        self.features_names = input_data.features_names

        if self.params.get('enable_categorical'):
            input_data = input_data.get_not_encoded_data()

        if self.params.get('use_eval_set'):
            train_input, eval_input = train_test_data_setup(input_data)

            X_train, y_train = self.convert_to_dataframe(
                train_input, identify_cats=self.params.get('enable_categorical')
            )

            X_eval, y_eval = self.convert_to_dataframe(
                eval_input, identify_cats=self.params.get('enable_categorical')
            )

            eval_metric = self.set_eval_metric(self.classes_)
            callbacks = self.update_callbacks()

            self.model.fit(
                X=X_train, y=y_train,
                eval_set=[(X_eval, y_eval)], eval_metric=eval_metric,
                callbacks=callbacks
            )

        else:
            X_train, y_train = self.convert_to_dataframe(
                input_data, identify_cats=self.params.get('enable_categorical')
            )

            self.model.fit(
                X=X_train, y=y_train,
            )

        return self.model

    def predict(self, input_data: InputData):
        if self.params.get('enable_categorical'):
            input_data = input_data.get_not_encoded_data()

        X, _ = self.convert_to_dataframe(input_data, identify_cats=self.params.get('enable_categorical'))
        prediction = self.model.predict(X)

        return prediction

    def check_and_update_params(self):
        early_stopping_rounds = self.params.get('early_stopping_rounds')
        use_eval_set = self.params.get('use_eval_set')

        if isinstance(early_stopping_rounds, int) and not use_eval_set:
            self.params.update(early_stopping_rounds=False)

    def update_callbacks(self) -> list:
        callback = []

        esr = self.params.get('early_stopping_rounds')
        if isinstance(esr, int):
            lgbm_early_stopping(esr, verbose=self.params.get('verbose'))

        return callback

    @staticmethod
    def set_eval_metric(n_classes):
        if n_classes is None:  # if n_classes is None -> regression
            eval_metric = ''

        elif len(n_classes) < 3:  # if n_classes < 3 -> bin class
            eval_metric = 'binary_logloss'

        else:  # else multiclass
            eval_metric = 'multi_logloss'

        return eval_metric

    @staticmethod
    def convert_to_dataframe(data: Optional[InputData], identify_cats: bool):
        dataframe = pd.DataFrame(data=data.features, columns=data.features_names)
        if data.target is not None and data.target.size > 0:
            dataframe['target'] = np.ravel(data.target)
        else:
            # TODO: temp workaround in case data.target is set to None intentionally
            #  for test.integration.models.test_model.check_predict_correct
            dataframe['target'] = np.zeros(len(data.features))

        if identify_cats and data.categorical_idx is not None:
            for col in dataframe.columns[data.categorical_idx]:
                dataframe[col] = dataframe[col].astype('category')

        if data.numerical_idx is not None:
            for col in dataframe.columns[data.numerical_idx]:
                dataframe[col] = dataframe[col].astype('float')

        return dataframe.drop(columns=['target']), dataframe['target']

    def plot_feature_importance(self):
        plot_feature_importance(self.features_names, self.model.feature_importances_)


class FedotLightGBMClassificationImplementation(FedotLightGBMImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.classes_ = None
        self.model = LGBMClassifier(**self.model_params)

    def fit(self, input_data: InputData):
        self.classes_ = np.unique(np.array(input_data.target))
        return super().fit(input_data=input_data)

    def predict_proba(self, input_data: InputData):
        if self.params.get('enable_categorical'):
            input_data = input_data.get_not_encoded_data()

        X, _ = self.convert_to_dataframe(input_data, self.params.get('enable_categorical'))
        prediction = self.model.predict_proba(X)
        return prediction


class FedotLightGBMRegressionImplementation(FedotLightGBMImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.classes_ = None
        self.model = LGBMRegressor(**self.model_params)


class FedotCatBoostImplementation(ModelImplementation):
    __operation_params = ['n_jobs', 'use_eval_set', 'enable_categorical']

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

        self.check_and_update_params()

        self.model_params = {k: v for k, v in self.params.to_dict().items() if k not in self.__operation_params}
        self.model = None
        self.features_names = None

    def fit(self, input_data: InputData):
        self.features_names = input_data.features_names

        if self.params.get('enable_categorical'):
            input_data = input_data.get_not_encoded_data()

        if self.params.get('use_eval_set'):
            # TODO: Using this method for tuning
            train_input, eval_input = train_test_data_setup(input_data)

            train_input = self.convert_to_pool(train_input, identify_cats=self.params.get('enable_categorical'))
            eval_input = self.convert_to_pool(eval_input, identify_cats=self.params.get('enable_categorical'))

            self.model.fit(X=train_input, eval_set=eval_input)

        else:
            train_input = self.convert_to_pool(input_data, identify_cats=self.params.get('enable_categorical'))

            self.model.fit(train_input)

        return self.model

    def predict(self, input_data: InputData):
        if self.params.get('enable_categorical'):
            input_data = input_data.get_not_encoded_data()

        prediction = self.model.predict(input_data.features)

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
    def convert_to_pool(data: Optional[InputData], identify_cats: bool):
        return Pool(
            data=data.features,
            label=data.target,
            cat_features=data.categorical_idx if identify_cats else None,
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
        plot_feature_importance(self.model.feature_names_, self.model.feature_importances_)


class FedotCatBoostClassificationImplementation(FedotCatBoostImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = CatBoostClassifier(**self.model_params)
        self.classes_ = None

    def fit(self, input_data: InputData):
        self.classes_ = np.unique(np.array(input_data.target))
        return super().fit(input_data=input_data)

    def predict_proba(self, input_data: InputData):
        if self.params.get('enable_categorical'):
            input_data = input_data.get_not_encoded_data()

        prediction = self.model.predict_proba(input_data.features)
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
