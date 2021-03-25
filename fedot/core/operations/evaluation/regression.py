import warnings

from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy, SkLearnEvaluationStrategy
from fedot.core.operations.evaluation.operation_implementations.\
    data_operations.sklearn_filters import LinearRegRANSAC, NonLinearRegRANSAC
from fedot.core.operations.evaluation.operation_implementations.\
    data_operations.sklearn_selectors import LinearRegFS, NonLinearRegFS
from fedot.core.repository.dataset_types import DataTypesEnum

warnings.filterwarnings("ignore", category=UserWarning)


class SkLearnRegressionStrategy(SkLearnEvaluationStrategy):
    def predict(self, trained_operation, predict_data: InputData,
                is_fit_chain_stage: bool):
        """
        Predict method for regression task
        :param trained_operation: model object
        :param predict_data: data used for prediction
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return:
        """

        prediction = trained_operation.predict(predict_data.features)
        # Wrap prediction as features for next level
        converted = OutputData(idx=predict_data.idx,
                               features=predict_data.features,
                               predict=prediction,
                               task=predict_data.task,
                               target=predict_data.target,
                               data_type=DataTypesEnum.table)

        return converted


class CustomRegressionPreprocessingStrategy(EvaluationStrategy):
    """ Strategy for applying custom algorithms from FEDOT to preprocess data
    for regression task
    """

    __operations_by_types = {
        'ransac_lin_reg': LinearRegRANSAC,
        'ransac_non_lin_reg': NonLinearRegRANSAC,
        'rfe_lin_reg': LinearRegFS,
        'rfe_non_lin_reg': NonLinearRegFS,
    }

    def __init__(self, operation_type: str, params: Optional[dict] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        """
        This method is used for operation training with the data provided
        :param InputData train_data: data used for operation training
        :return: trained data operation
        """

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if self.params_for_fit:
            operation_implementation = self.operation_impl(**self.params_for_fit)
        else:
            operation_implementation = self.operation_impl()

        operation_implementation.fit(train_data)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_chain_stage: bool):
        """
        Transform method for preprocessing

        :param trained_operation: model object
        :param predict_data: data used for prediction
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return:
        """
        # Prediction here is already OutputData type object
        prediction = trained_operation.transform(predict_data, is_fit_chain_stage)
        return prediction

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain Custom Regression Preprocessing Strategy for {operation_type}')

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))
