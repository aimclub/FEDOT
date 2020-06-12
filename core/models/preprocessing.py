import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

from core.repository.dataset_types import DataTypesEnum


class PreprocessingStrategy:
    def fit(self, data_to_fit) -> 'PreprocessingStrategy':
        raise NotImplementedError()

    def apply(self, data):
        raise NotImplementedError()


class Scaling(PreprocessingStrategy):
    def __init__(self, with_imputation=True):
        if with_imputation:
            self.default = ImputationStrategy()
        self.with_imputation = with_imputation
        self.scaler = preprocessing.StandardScaler()

    def fit(self, data_to_fit):
        if self.with_imputation:
            self.default.fit(data_to_fit)
            data_to_fit = self.default.apply(data_to_fit)
        self.scaler.fit(data_to_fit)
        return self

    def apply(self, data):
        if self.with_imputation:
            data = self.default.apply(data)
        resulted = self.scaler.transform(data)
        return resulted


class Normalization(PreprocessingStrategy):
    def __init__(self):
        self.default = ImputationStrategy()

    def fit(self, data_to_fit):
        self.default.fit(data_to_fit)
        return self

    def apply(self, data):
        modified = self.default.apply(data)
        resulted = preprocessing.normalize(modified)

        return resulted


class ImputationStrategy(PreprocessingStrategy):
    def __init__(self):
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    def fit(self, data_to_fit):
        self.imputer.fit(data_to_fit)
        return self

    def apply(self, data):
        modified = self.imputer.transform(data)
        return modified


class EmptyStrategy(PreprocessingStrategy):
    def fit(self, data_to_fit):
        return self

    def apply(self, data):
        return data


class LaggedFeatureScalingStrategy(Scaling):
    def __init__(self):
        super().__init__(with_imputation=False)


class LaggedFeature3dScalingStrategy(PreprocessingStrategy):
    def __init__(self):
        self.scaling = Scaling(with_imputation=False)

    def fit(self, data_to_fit):
        # Make from (n, timestamps, features) input the (n, features)
        if data_to_fit.ndim == 3:
            data_to_fit = data_to_fit[:, -1]
        self.scaling.fit(data_to_fit)
        return self

    def apply(self, data):
        # Make (n * timestamps, features) from (n, timestamps, features)
        # So each feature scaled separatedly
        if data.ndim == 2:
            return self.scaling.apply(data)
        temp = data.reshape(-1, data.shape[-1])
        scaled = self.scaling.apply(temp)
        resulted = scaled.reshape(data.shape)
        return resulted


_preprocessing_for_input_data = {
    DataTypesEnum.ts: EmptyStrategy,
    DataTypesEnum.table: Scaling,
    DataTypesEnum.ts_lagged_table: LaggedFeatureScalingStrategy,
    DataTypesEnum.ts_lagged_3d: LaggedFeature3dScalingStrategy,
}


def preprocessing_func_for_data(data: 'InputData', node: 'Node'):
    preprocessing_func = EmptyStrategy
    if 'without_preprocessing' not in node.model.metadata.tags:
        if node.manual_preprocessing_func:
            preprocessing_func = node.manual_preprocessing_func
        else:
            preprocessing_func = _preprocessing_for_input_data[data.data_type]
    return preprocessing_func
