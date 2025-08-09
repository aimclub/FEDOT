import os

import numpy as np
from tabpfn import TabPFNClassifier, TabPFNRegressor
from typing import Optional
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.utils import default_fedot_data_dir


class FedotTabPFNImplementation(ModelImplementation):
    __operation_params = [
        'enable_categorical',
        'max_samples',
        'max_features',
        'model_path'
    ]

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

        self.model_params = {
            k: v for k, v in self.params.to_dict().items() if k not in self.__operation_params
        }

        model_path = self.params.get('model_path', None)

        if model_path == "auto":
            self.model_params['model_path'] = os.path.join(default_fedot_data_dir(), 'tabpfn')
            model_path = os.path.join(default_fedot_data_dir(), 'tabpfn')
            if not os.path.exists(model_path):
                os.makedirs(model_path, exist_ok=True)
            os.environ["TABPFN_MODEL_CACHE_DIR"] = model_path
        elif model_path is not None:
            self.model_params['model_path'] = model_path
            os.environ["TABPFN_MODEL_CACHE_DIR"] = model_path

        self.model = None
        self.classes_ = None

    def fit(self, input_data: InputData):
        self.model.categorical_features_indices = input_data.categorical_idx

        if self.params.get('enable_categorical'):
            input_data = input_data.get_not_encoded_data()

        self.model.fit(X=input_data.features, y=input_data.target)

        return self.model

    def predict(self, input_data: InputData) -> OutputData:
        if self.params.get('enable_categorical'):
            input_data = input_data.get_not_encoded_data()

        prediction = self.model.predict(input_data.features)

        output_data = self._convert_to_output(
            input_data=input_data,
            predict=prediction
        )
        return output_data

    def predict_proba(self, input_data: InputData):
        if self.params.get('enable_categorical'):
            input_data = input_data.get_not_encoded_data()

        prediction = self.model.predict_proba(input_data.features)
        output_data = self._convert_to_output(
            input_data=input_data,
            predict=prediction
        )
        return output_data


class FedotTabPFNClassificationImplementation(FedotTabPFNImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = TabPFNClassifier(**self.model_params)

    def fit(self, input_data: InputData):
        self.classes_ = np.unique(np.array(input_data.target))
        return super().fit(input_data=input_data)


class FedotTabPFNRegressionImplementation(FedotTabPFNImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = TabPFNRegressor(**self.model_params)
