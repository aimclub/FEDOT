import os
from typing import Optional

import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.utils import default_fedot_data_dir


class BaseFoundationalImplementation(ModelImplementation):
    _excluded_model_params = [
        'max_samples',
        'max_features',
        'device'
    ]

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model_params = {
            k: v for k, v in self.params.to_dict().items() if k not in self._excluded_model_params
        }
        self.model = None
        self.classes_ = None

    def _preprocess_fit_data(self, input_data: InputData) -> InputData:
        return input_data

    def _preprocess_predict_data(self, input_data: InputData) -> InputData:
        return input_data

    def fit(self, input_data: InputData):
        input_data = self._preprocess_fit_data(input_data)
        self.model.fit(X=input_data.features, y=input_data.target)
        return self.model

    def predict(self, input_data: InputData) -> OutputData:
        input_data = self._preprocess_predict_data(input_data)
        prediction = self.model.predict(input_data.features)
        return self._convert_to_output(
            input_data=input_data,
            predict=prediction
        )

    def predict_proba(self, input_data: InputData):
        input_data = self._preprocess_predict_data(input_data)
        prediction = self.model.predict_proba(input_data.features)
        return self._convert_to_output(
            input_data=input_data,
            predict=prediction
        )


class BaseFoundationalClassificationImplementation(BaseFoundationalImplementation):
    def fit(self, input_data: InputData):
        self.classes_ = np.unique(np.array(input_data.target))
        return super().fit(input_data=input_data)


class FedotTabPFNImplementation(BaseFoundationalImplementation):
    """FEDOT implementation of TabPFN that uses tabpfn model version 2.
    To use the newer TabPFN v2.5 model, you must accept the non-commercial usage license
    in your Hugging Face account at https://huggingface.co/Prior-Labs/tabpfn_2_5.
    After accepting the license, update the `tabpfn` library to the latest version
    (currently 6.0.6) and log in to your Hugging Face account, for example using env variable:
    `os.environ["HF_TOKEN"] = "your_hf_token"`. The library will then automatically
    use the v2.5 model. If authentication with a token fails, you will get an error.

    """

    _excluded_model_params = [
        'enable_categorical',
        'max_samples',
        'max_features',
        'model_path'
    ]

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

        model_path = self.params.get('model_path', None)
        if model_path == "auto":
            model_path = os.path.join(default_fedot_data_dir(), 'tabpfn')
            if not os.path.exists(model_path):
                os.makedirs(model_path, exist_ok=True)
            os.environ["TABPFN_MODEL_CACHE_DIR"] = model_path
        elif model_path is not None:
            os.environ["TABPFN_MODEL_CACHE_DIR"] = model_path

    def _preprocess_fit_data(self, input_data: InputData) -> InputData:
        self.model.categorical_features_indices = input_data.categorical_idx
        if self.params.get('enable_categorical'):
            return input_data.get_not_encoded_data()
        return input_data

    def _preprocess_predict_data(self, input_data: InputData) -> InputData:
        if self.params.get('enable_categorical'):
            return input_data.get_not_encoded_data()
        return input_data


class FedotTabPFNClassificationImplementation(
    FedotTabPFNImplementation,
    BaseFoundationalClassificationImplementation
):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        from tabpfn import TabPFNClassifier
        self.model = TabPFNClassifier(**self.model_params)


class FedotTabPFNRegressionImplementation(FedotTabPFNImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        from tabpfn import TabPFNRegressor
        self.model = TabPFNRegressor(**self.model_params)


class FedotTabICLImplementation(BaseFoundationalImplementation):
    _excluded_model_params = [
        'max_samples',
        'max_features'
    ]


class FedotTabICLClassificationImplementation(
    FedotTabICLImplementation,
    BaseFoundationalClassificationImplementation
):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        from tabicl import TabICLClassifier
        self.model = TabICLClassifier(**self.model_params)


class FedotTabICLRegressionImplementation(FedotTabICLImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        from tabicl import TabICLRegressor
        self.model = TabICLRegressor(**self.model_params)
