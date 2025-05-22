import numpy as np
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier, AutoTabPFNRegressor
from typing import Optional
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class FedotTabPFNImplementation(ModelImplementation):
    __operation_params = [
        'enable_categorical',
        'max_samples_cpu',
        'max_samples_gpu',
    ]

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

        self.model_params = {
            k: v for k, v in self.params.to_dict().items() if k not in self.__operation_params
        }
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


class FedotAutoTabPFNClassificationImplementation(FedotTabPFNImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = AutoTabPFNClassifier(**self.model_params)

    def fit(self, input_data: InputData):
        self.classes_ = np.unique(np.array(input_data.target))
        return super().fit(input_data=input_data)


class FedotAutoTabPFNRegressionImplementation(FedotTabPFNImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = AutoTabPFNRegressor(**self.model_params)
