import warnings
from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import \
    ExogDataTransformationImplementation, GaussianFilterImplementation, LaggedTransformationImplementation, \
    TsSmoothingImplementation, SparseLaggedTransformationImplementation, CutImplementation, \
    NumericalDerivativeFilterImplementation
from fedot.core.operations.evaluation.operation_implementations.models.ts_implementations.naive import \
    RepeatLastValueImplementation, NaiveAverageForecastImplementation
from fedot.core.operations.evaluation.operation_implementations.models. \
    ts_implementations.statsmodels import AutoRegImplementation, GLMImplementation, ExpSmoothingImplementation
from fedot.core.operations.evaluation.operation_implementations.models.ts_implementations.arima import \
    ARIMAImplementation, STLForecastARIMAImplementation
from fedot.core.operations.evaluation.operation_implementations.models.ts_implementations.clstm import \
    CLSTMImplementation
from fedot.core.operations.evaluation.operation_implementations.models.ts_implementations.poly import \
    PolyfitImplementation
from fedot.core.operations.operation_parameters import OperationParameters

warnings.filterwarnings("ignore", category=UserWarning)


class FedotTsForecastingStrategy(EvaluationStrategy):
    """
    This class defines the certain classical models implementation for time
    series forecasting (e.g. AR, ARIMA)

    :param str operation_type: str type of the operation defined in operation or
    data operation repositories
    :param dict params: hyperparameters to fit the model with
    """

    __operations_by_types = {
        'arima': ARIMAImplementation,
        'ar': AutoRegImplementation,
        'stl_arima': STLForecastARIMAImplementation,
        'ets': ExpSmoothingImplementation,
        'clstm': CLSTMImplementation,
        'polyfit': PolyfitImplementation,
        'glm': GLMImplementation,
        'locf': RepeatLastValueImplementation,
        'ts_naive_average': NaiveAverageForecastImplementation
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation = self._convert_to_operation(operation_type)

    def fit(self, train_data: InputData):
        """
        This method is used for operation training with the data provided
        :param InputData train_data: data used for operation training
        :return: trained model
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        model = self.operation(self.params_for_fit)

        model.fit(train_data)
        return model

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        This method used for prediction of the target data during predict stage.

        :param trained_operation: trained operation object
        :param predict_data: data to predict
        :return OutputData: passed data with new predicted target
        """

        prediction = trained_operation.predict(predict_data)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def predict_for_fit(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        This method used for prediction of the target data during fit stage.

        :param trained_operation: trained operation object
        :param predict_data: data to predict
        :return OutputData: passed data with new predicted target
        """

        prediction = trained_operation.predict_for_fit(predict_data)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain custom time series forecasting '
                             f'strategy for {operation_type}')


class FedotTsTransformingStrategy(EvaluationStrategy):
    """
    This class defines the certain data operation implementation for time series
    forecasting

    :param str operation_type: str type of the operation defined in operation or
    data operation repositories
    :param dict params: hyperparameters to fit the model with
    """

    __operations_by_types = {
        'lagged': LaggedTransformationImplementation,
        'sparse_lagged': SparseLaggedTransformationImplementation,
        'smoothing': TsSmoothingImplementation,
        'exog_ts': ExogDataTransformationImplementation,
        'gaussian_filter': GaussianFilterImplementation,
        'diff_filter': NumericalDerivativeFilterImplementation,
        'cut': CutImplementation}

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation = self._convert_to_operation(self.operation_type)

    def fit(self, train_data: InputData):
        """
        This method is used for operation training with the data provided
        :param InputData train_data: data used for operation training
        :return: trained operation (if it is needed for applying)
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        transformation_operation = self.operation(self.params_for_fit)
        transformation_operation.fit(train_data)
        return transformation_operation

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        This method used for prediction of the target data during predict stage.

        :param trained_operation: trained operation object
        :param predict_data: data to predict
        :return OutputData: passed data with new predicted target
        """

        prediction = trained_operation.transform(predict_data)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def predict_for_fit(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        This method used for prediction of the target data during fit stage.

        :param trained_operation: trained operation object
        :param predict_data: data to predict
        :return OutputData: passed data with new predicted target
        """

        prediction = trained_operation.transform_for_fit(predict_data)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain custom time series transforming '
                             f'strategy for {operation_type}')
