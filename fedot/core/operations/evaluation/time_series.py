import warnings

from typing import Optional

from fedot.core.operations.evaluation.operation_implementations.models.\
    ts_implementations import ARIMAImplementation, AutoRegImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations \
    import LaggedTransformationImplementation, TsSmoothingImplementation, \
    ExogDataTransformationImplementation, GaussianFilterImplementation

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy

warnings.filterwarnings("ignore", category=UserWarning)


class CustomTsForecastingStrategy(EvaluationStrategy):
    """
    This class defines the certain classical models implementation for time
    series forecasting (e.g. AR, ARIMA)

    :param str operation_type: str type of the operation defined in operation or
    data operation repositories
    :param dict params: hyperparameters to fit the model with
    """

    __operations_by_types = {
        'arima': ARIMAImplementation,
        'ar': AutoRegImplementation}

    def __init__(self, operation_type: str, params: Optional[dict] = None):
        super().__init__(operation_type, params)
        self.operation = self._convert_to_operation(operation_type)
        self.params_for_fit = params

    def fit(self, train_data: InputData):
        """
        This method is used for operation training with the data provided
        :param InputData train_data: data used for operation training
        :return: trained model
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if self.params_for_fit:
            model = self.operation(**self.params_for_fit)
        else:
            model = self.operation()

        model.fit(train_data)
        return model

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_chain_stage: bool) -> OutputData:
        """
        This method used for prediction of the target data.

        :param trained_operation: trained operation object
        :param predict_data: data to predict
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return OutputData: passed data with new predicted target
        """

        prediction = trained_operation.predict(predict_data,
                                               is_fit_chain_stage)
        # Convert prediction to output (if it is required)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain custom time series forecasting '
                             f'strategy for {operation_type}')


class CustomTsTransformingStrategy(EvaluationStrategy):
    """
    This class defines the certain data operation implementation for time series
    forecasting

    :param str operation_type: str type of the operation defined in operation or
    data operation repositories
    :param dict params: hyperparameters to fit the model with
    """

    __operations_by_types = {
        'lagged': LaggedTransformationImplementation,
        'smoothing': TsSmoothingImplementation,
        'exog': ExogDataTransformationImplementation,
        'gaussian_filter': GaussianFilterImplementation}

    def __init__(self, operation_type: str, params: Optional[dict] = None):
        super().__init__(operation_type, params)
        self.operation = self._convert_to_operation(operation_type)
        self.params_for_fit = params

    def fit(self, train_data: InputData):
        """
        This method is used for operation training with the data provided
        :param InputData train_data: data used for operation training
        :return: trained operation (if it is needed for applying)
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if self.params_for_fit:
            transformation_operation = self.operation(**self.params_for_fit)
        else:
            transformation_operation = self.operation()

        transformation_operation.fit(train_data)
        return transformation_operation

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_chain_stage: bool) -> OutputData:
        """
        This method used for prediction of the target data.

        :param trained_operation: trained operation object
        :param predict_data: data to predict
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return OutputData: passed data with new predicted target
        """

        prediction = trained_operation.transform(predict_data,
                                                 is_fit_chain_stage)
        # Convert prediction to output (if it is required)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain custom time series transforming '
                             f'strategy for {operation_type}')
