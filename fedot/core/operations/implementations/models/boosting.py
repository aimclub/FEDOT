import os
from typing import Optional

import numpy as np
import torch
from catboost import CatBoostClassifier, CatBoostRegressor, FeaturesData, Pool
from lightgbm import LGBMClassifier, LGBMRegressor
from matplotlib import pyplot as plt
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from xgboost import XGBClassifier, XGBRegressor

from fedot.core.backend.backend import Backend
from fedot.core.backend.array_conversions import as_numpy, as_torch
from fedot.core.data.prepared_data.prepared_data import PreparedData
from fedot.core.data.split.data_split import train_test_tensor_data_setup
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.implementations.base import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.utils import default_fedot_data_dir, is_multi_output_target
from fedot.preprocessing.methods.categorical_encoding import LabelEncoder


class BoostingImplementation(ModelImplementation):
    """Shared TensorData preparation; numerical work stays in Torch until the library boundary."""

    _operation_params = ('n_jobs', 'use_eval_set', 'enable_categorical')

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model_params = {key: value for key, value in self.params.to_dict().items()
                             if key not in self._operation_params}
        self.model = None
        self.features_names = None
        self.classes_ = None
        self._categorical_idx = []
        self._category_encoder = LabelEncoder()

    def _training_data(self, data: TensorData):
        self.features_names = data.features_names
        self.classes_ = torch.unique(data.target) if data.task.task_type == TaskTypesEnum.classification else None
        if not self.params.get('use_eval_set') or is_multi_output_target(data):
            return data, None
        train, evaluation = train_test_tensor_data_setup(data, stratify=True)
        if (data.task.task_type == TaskTypesEnum.classification
                and not torch.isin(torch.unique(train.target), torch.unique(evaluation.target)).all()):
            return data, None
        return train, evaluation

    def prepare_categorical_features(self, features, *, fit: bool = False):
        """Use the source encoder's vocabulary; unknown categories remain missing for the library."""
        prepared = PreparedData(features=features.copy())
        # Original strings live in a host sidecar, independent of the feature tensor device.
        with Backend().override('cpu'):
            if fit:
                self._category_encoder.fit(prepared, list(range(features.shape[1])))
            encoded = self._category_encoder.transform(prepared).features.astype(np.float32)
        for column, unknown in self._category_encoder.unknown_labels_.items():
            encoded[encoded[:, column] == unknown, column] = np.nan
        return encoded

    def _feature_tensor(self, data: TensorData, *, fit: bool = False):
        if not self.params.get('enable_categorical'):
            if fit:
                self._categorical_idx = []
            return data.features
        original = data.get_not_encoded_data()
        width = original.numerical_features.shape[1]
        if fit:
            self.features_names = original.features_names
            self._categorical_idx = list(range(width, width + original.categorical_features.shape[1]))
        if original.categorical_features.shape[1] == 0:
            return original.numerical_features
        encoded = self.prepare_categorical_features(original.categorical_features, fit=fit)
        categories = as_torch(encoded, device=data.device, dtype=data.features.dtype)
        return torch.cat((original.numerical_features, categories), dim=1)

    @staticmethod
    def _target_tensor(data: TensorData):
        target = data.target
        return target.reshape(-1) if target.ndim == 2 and target.shape[1] == 1 else target

    def _prepare_target(self, data: TensorData):
        return as_numpy(self._target_tensor(data))

    @staticmethod
    def _prepare_prediction(prediction, data: TensorData):
        if isinstance(prediction, list):
            return torch.stack([as_torch(part, device=data.device) for part in prediction], dim=1)
        return as_torch(prediction, device=data.device)

    def predict(self, data: TensorData) -> torch.Tensor:
        return self._prepare_prediction(self.model.predict(self._prepare_features(data)), data)

    def predict_proba(self, data: TensorData) -> torch.Tensor:
        return self._prepare_prediction(self.model.predict_proba(self._prepare_features(data)), data)


class FedotXGBoostImplementation(BoostingImplementation):
    _operation_params = ('n_jobs', 'use_eval_set')

    @property
    def _model_device(self) -> torch.device:
        return torch.device((self.model.get_params().get('device') or 'cpu').replace('gpu', 'cuda'))

    def _prepare_features(self, data: TensorData, *, fit: bool = False):
        features = self._feature_tensor(data, fit=fit)
        if fit:
            types = ['c' if pos in self._categorical_idx else 'q' for pos in range(features.shape[1])]
            self.model.set_params(feature_types=types)
        return as_torch(features, device=self._model_device, contiguous=True)

    def _prepare_target(self, data: TensorData):
        return as_torch(self._target_tensor(data), device=self._model_device, contiguous=True)

    def _predict(self, data: TensorData, *, probabilities: bool) -> torch.Tensor:
        features = self._prepare_features(data)
        if not features.is_cuda or self.model.booster == 'gblinear':
            method = self.model.predict_proba if probabilities else self.model.predict
            return self._prepare_prediction(method(features), data)

        booster = self.model.get_booster()
        best_iteration = booster.attr('best_iteration')
        iterations = (0, int(best_iteration) + 1) if best_iteration is not None else (0, 0)
        softmax = probabilities and self.model.objective == 'multi:softmax'
        prediction = booster.inplace_predict(features, iteration_range=iterations, missing=self.model.missing,
                                             predict_type='margin' if softmax else 'value')
        prediction = as_torch(prediction)
        if self.classes_ is not None:
            if probabilities:
                if softmax:
                    prediction = prediction.softmax(dim=1)
                elif prediction.ndim == 1:
                    prediction = torch.stack((1 - prediction, prediction), dim=1)
            elif prediction.ndim > 1 and self.model.n_classes_ != 2:
                prediction = prediction.argmax(dim=1)
            elif prediction.ndim > 1 and prediction.shape[1] != 1:
                # Match sklearn's multilabel output, including its float64 dtype.
                prediction = (prediction > .5).to(torch.float64)
            elif self.model.objective == 'multi:softmax':
                prediction = prediction.to(torch.int32)
            else:
                prediction = (prediction > .5).to(torch.int64)
        return as_torch(prediction, device=data.device)

    def predict(self, data: TensorData) -> torch.Tensor:
        return self._predict(data, probabilities=False)

    def predict_proba(self, data: TensorData) -> torch.Tensor:
        return self._predict(data, probabilities=True)

    def fit(self, data: TensorData):
        train, evaluation = self._training_data(data)
        features = self._prepare_features(train, fit=True)
        target = self._prepare_target(train)
        if evaluation is not None:
            self.model.eval_metric = self.set_eval_metric(self.classes_)
            eval_set = [(self._prepare_features(evaluation), self._prepare_target(evaluation))]
        else:
            self.model.early_stopping_rounds = None
            self.params.update(early_stopping_rounds=None)
            eval_set = None
        self.model.fit(features, target, eval_set=eval_set, verbose=self.model_params['verbosity'])
        return self.model

    @staticmethod
    def set_eval_metric(classes):
        if classes is None:
            return 'rmse'
        return 'auc' if len(classes) < 3 else 'mlogloss'

    def get_feature_importance(self) -> list:
        return self.model.feature_importances_

    def plot_feature_importance(self, importance_type='weight'):
        scores = self.model.get_booster().get_score(importance_type=importance_type)
        plot_feature_importance(list(scores), list(scores.values()))


class FedotXGBoostClassificationImplementation(FedotXGBoostImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = XGBClassifier(**self.model_params)


class FedotXGBoostRegressionImplementation(FedotXGBoostImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = XGBRegressor(**self.model_params)


class FedotLightGBMImplementation(BoostingImplementation):
    def _prepare_features(self, data: TensorData, *, fit: bool = False):
        return as_numpy(self._feature_tensor(data, fit=fit))

    def fit(self, data: TensorData):
        train, evaluation = self._training_data(data)
        features = self._prepare_features(train, fit=True)
        target = self._prepare_target(train)
        if evaluation is not None:
            self.model.fit(features, target,
                           eval_set=[(self._prepare_features(evaluation), self._prepare_target(evaluation))],
                           eval_metric=self.set_eval_metric(self.classes_), categorical_feature=self._categorical_idx)
        else:
            self.model.set_params(early_stopping_rounds=None)
            self.params.update(early_stopping_rounds=None)
            if is_multi_output_target(data):
                wrapper = (MultiOutputClassifier if data.task.task_type == TaskTypesEnum.classification
                           else MultiOutputRegressor)
                self.model = wrapper(self.model)
            self.model.fit(features, target, categorical_feature=self._categorical_idx)
        return self.model

    @staticmethod
    def set_eval_metric(classes):
        if classes is None:
            return ''
        return 'binary_logloss' if len(classes) < 3 else 'multi_logloss'

    def plot_feature_importance(self):
        plot_feature_importance(self.features_names, self.model.feature_importances_)


class FedotLightGBMClassificationImplementation(FedotLightGBMImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = LGBMClassifier(**self.model_params)


class FedotLightGBMRegressionImplementation(FedotLightGBMImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = LGBMRegressor(**self.model_params)


class FedotCatBoostImplementation(BoostingImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.params.update(thread_count=self.params.get('n_jobs'))
        self.model_params['thread_count'] = self.params.get('n_jobs')

    def prepare_categorical_features(self, features, *, fit: bool = False):
        """CatBoost requires strings/integers; normalized missing categories become the string nan."""
        return features.astype(str).astype(object)

    def _prepare_features(self, data: TensorData, *, with_target: bool = False):
        if self.params.get('enable_categorical'):
            original = data.get_not_encoded_data()
            numerical = as_numpy(original.numerical_features, dtype=np.float32)
            categories = self.prepare_categorical_features(original.categorical_features)
            width = numerical.shape[1]
            names = original.features_names
            features = FeaturesData(num_feature_data=numerical if numerical.shape[1] else None,
                                    cat_feature_data=categories if categories.shape[1] else None,
                                    num_feature_names=names[:width] if names is not None else None,
                                    cat_feature_names=names[width:] if names is not None else None)
            names = None
        else:
            features = as_numpy(data.features)
            names = data.features_names
        return Pool(features, label=self._prepare_target(data) if with_target else None,
                    feature_names=names)

    def fit(self, data: TensorData):
        train, evaluation = self._training_data(data)
        train_pool = self._prepare_features(train, with_target=True)
        self.features_names = train_pool.get_feature_names()
        if evaluation is not None:
            self.model.fit(train_pool, eval_set=self._prepare_features(evaluation, with_target=True))
        else:
            for name in ('early_stopping_rounds', 'od_wait', 'od_type', 'od_pval'):
                self.model._init_params.pop(name, None)
            self.model._init_params['use_best_model'] = False
            self.params.update(use_best_model=False)
            self.model.fit(train_pool)
        return self.model

    def save_model(self, model_name: str = 'catboost'):
        save_path = os.path.join(default_fedot_data_dir(), f'catboost/{model_name}.cbm')
        self.model.save_model(save_path, format='cbm')

    def load_model(self, path):
        self.model = type(self.model)()
        self.model.load_model(path)

    def get_feature_importance(self):
        return self.model.get_feature_importance(prettified=True)

    def plot_feature_importance(self):
        plot_feature_importance(self.model.feature_names_, self.model.feature_importances_)


class FedotCatBoostClassificationImplementation(FedotCatBoostImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = CatBoostClassifier(**self.model_params)


class FedotCatBoostRegressionImplementation(FedotCatBoostImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = CatBoostRegressor(**self.model_params)


def plot_feature_importance(feature_names, feature_importance):
    importance = np.asarray(list(feature_importance))
    names = np.asarray(feature_names if feature_names is not None else np.arange(len(importance)))
    selected = np.flatnonzero(importance > .1)
    selected = selected[np.argsort(importance[selected])]
    plt.figure(figsize=(16, 9))
    plt.barh(names[selected], importance[selected])
    plt.title('Feature Importance')
    plt.show()
