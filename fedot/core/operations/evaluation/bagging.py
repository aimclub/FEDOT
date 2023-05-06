from abc import ABC
from typing import Optional

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from deslib.dcs.mcb import MCB
from deslib.dcs.ola import OLA
from deslib.des.des_p import DESP
from deslib.des.knora_e import KNORAE
from deslib.des.knora_u import KNORAU
from deslib.des.meta_des import METADES

from fedot.core.data.data import InputData, OutputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.operation_parameters import OperationParameters, get_default_params


class SkLearnBaggingStrategy(EvaluationStrategy, ABC):
    """ FEDOT's bagging base class. Set params to bagging class

        Args:
            operation_type: 'str' selected operation as a base model in bagging

            .. details:: possible bagging operations:

                - ``bag_dt`` -> Bagging of the Decision Tree
                - ``bag_catboost`` -> Bagging of the Catboost classifier
                - ``bag_xgboost`` -> Bagging of the Xgboost classifier
                - ``bag_lgbm`` -> Bagging of the LightGBM
                - ``bag_lgbmxt`` -> Bagging of the LightGBM Extra Trees

                - ``bag_dtreg`` -> Bagging of the Decision Trees Regressors
                - ``bag_adareg`` -> AdaBoosting Regressor
                - ``bag_xgboostreg`` -> Bagging of the Xgboost classifier
                - ``bag_catboostreg`` -> Bagging of the Catboost classifier
                - ``bag_lgbmreg`` -> Bagging of the LightGBM
                - ``bag_lgbmxtreg`` -> Bagging of the LightGBM Extra Trees

            params: operation's init and fitting hyperparameters

            .. details:: explanation of params
                - ``bagging_params`` - params belong to bagging operation:
                    - ``n_estimators`` - numbers of base learner in bagging structure
                    - ``bootstrap`` - whether samples are drawn with replacement. If False, sampling without replacement
                                      is performed.
                    - ``oob_score`` - whether to use out-of-bag samples to estimate the generalization error.
                                      Only available if bootstrap=True.
                    - ``n_jobs`` - the number of jobs to run in parallel

                - ``model_params`` - model's fitting params, which is used in bagging


    """

    _operations_by_types = {
        'bag_dtreg': DecisionTreeRegressor,
        'bag_adareg': AdaBoostRegressor,
        'bag_xgboostreg': XGBRegressor,
        'bag_catboostreg': CatBoostRegressor,
        'bag_lgbmreg': LGBMRegressor,
        'bag_lgbmxtreg': LGBMRegressor,

        'bag_dt': DecisionTreeClassifier,
        'bag_catboost': CatBoostClassifier,
        'bag_xgboost': XGBClassifier,
        'bag_lgbm': LGBMClassifier,
        'bag_lgbmxt': LGBMClassifier
    }

    _ds_techniques_by_types = {
        'knora': KNORAU,
        'kne': KNORAE,
        'desp': DESP,
        'ola': OLA,
        'mcb': MCB,
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation_impl = None
        self.bagging_operation = None
        self._ds_technique = None

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self._operations_by_types.keys():
            if self._model_params:
                self._bagging_params['estimator'] = self._operations_by_types[operation_type](**self._model_params)
            else:
                self._bagging_params['estimator'] = self._operations_by_types[operation_type]()

            return self.bagging_operation(**self._bagging_params)

        else:
            raise ValueError(f'Impossible to create bagging operation for {operation_type}')

    def _set_operation_params(self, operation_type, params):
        if params is None:
            params = get_default_params(operation_type)
        elif isinstance(params, dict):
            params = OperationParameters.from_operation_type(operation_type, **params)
        elif isinstance(params, OperationParameters):
            if params.get('model_params'):
                params = OperationParameters.from_operation_type(operation_type, **(params.to_dict()))

        self._model_params = params.get('model_params')
        # TODO: sklearn param base_estimator will change to estimator in future since 1.4

        self._bagging_params = {}

        for param in params.keys():
            if param != 'model_params' and param != 'dynamic_selection_technique':
                self._bagging_params.update({param: params.get(param)})

        if params.get('dynamic_selection_technique'):
            self._ds_technique = params.get('dynamic_selection_technique')

        return params

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))

    def fit(self, train_data: InputData):
        """ Method to train chosen operation with provided data

        Args:
            train_data: data used for operation training

        Returns:
            trained bagging model
        """
        operation_implementation = self.operation_impl

        if self._ds_technique:
            train_data, dsel_data = train_test_data_setup(train_data, split_ratio=0.5)

        # TODO: Fix RandomStateHandler()
        # with RandomStateHandler():
        #     operation_implementation.fit(train_data.features, train_data.target)
        operation_implementation.fit(train_data.features, train_data.target)

        if self._ds_technique:
            dynamic_selection = self._ds_techniques_by_types[self._ds_technique](operation_implementation)
            dynamic_selection.fit(dsel_data.features, dsel_data.target)

            return dynamic_selection

        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData):
        """ Method to predict

        Args:
            trained_operation: operation object
            predict_data: data to predict

        Returns:
            passed data with new predicted target
        """
        NotImplementedError()


class SkLearnBaggingClassificationStrategy(SkLearnBaggingStrategy):
    """ FEDOT's classification bagging operation implementation from sklearn

            Args:
                operation_type: 'str' selected operation as a base model in bagging
                params: operation's init and fitting hyperparameters
    """

    def __init__(self, operation_type, params: Optional[OperationParameters] = None):
        params = self._set_operation_params(operation_type, params)
        super().__init__(operation_type, params)
        self.bagging_operation = BaggingClassifier
        self.operation_impl = self._convert_to_operation(operation_type)

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        if self.output_mode in ['default', 'labels']:
            prediction = trained_operation.predict(predict_data.features)
        elif self.output_mode in ['probs', 'full_probs'] and predict_data.task:
            prediction = trained_operation.predict_proba(predict_data.features)
        else:
            raise ValueError(f'Output model {self.output_mode} is not supported')

        return self._convert_to_output(prediction, predict_data)


class SkLearnBaggingRegressionStrategy(SkLearnBaggingStrategy):
    """ FEDOT's regression bagging operation implementation from sklearn

            Args:
                operation_type: 'str' selected operation as a base model in bagging
                params: operation's init and fitting hyperparameters
    """

    def __init__(self, operation_type, params: Optional[OperationParameters] = None):
        params = self._set_operation_params(operation_type, params)
        super().__init__(operation_type, params)
        self.bagging_operation = BaggingRegressor
        self.operation_impl = self._convert_to_operation(operation_type)

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        prediction = trained_operation.predict(predict_data.features)

        return self._convert_to_output(prediction, predict_data)
