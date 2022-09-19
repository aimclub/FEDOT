import os
from typing import Optional

import numpy as np

from h2o import h2o, H2OFrame
from h2o.automl import H2OAutoML
from tpot import TPOTClassifier, TPOTRegressor

from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.automl_wrappers import *

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.repository.tasks import TaskTypesEnum


class H2OAutoMLRegressionStrategy(EvaluationStrategy):
    __operations_by_types = {
        'h2o_regr': H2OAutoML
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        ip, port = self._get_h2o_connect_config()
        h2o.init(ip=ip, port=port, name='h2o_server')

        frame = self._data_transform(train_data)

        train_frame, valid_frame = frame.split_frame(ratios=[0.85])

        # make sure that your target column is the last one
        train_columns = train_frame.columns
        if train_data.task.task_type == TaskTypesEnum.ts_forecasting:
            target_len = train_data.task.task_params.forecast_length
        else:
            target_len = 1
        target_names = train_columns[-target_len:]

        models = []

        for name in target_names:
            train_columns.remove(name)
        for name in target_names:
            model = H2OAutoML(max_models=self.params_for_fit.get("max_models"),
                              seed=self.params_for_fit.get("seed"),
                              max_runtime_secs=self.params_for_fit.get("timeout") * 60 // target_len
                              )
            model.train(x=train_columns, y=name, training_frame=train_frame)
            models.append(model.leader)

        return H2OSerializationWrapper(models)

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        res = []
        for model in trained_operation.get_estimators():
            frame = H2OFrame(predict_data.features)
            prediction = model.predict(frame)
            prediction = prediction.as_data_frame().to_numpy()
            res.append(np.ravel(prediction))
        res = np.hstack(res)
        out = self._convert_to_output(res, predict_data)
        return out

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain H2O AutoML Regression Strategy for {operation_type}')

    def _data_transform(self, data: InputData) -> H2OFrame:
        if len(data.target.shape) == 1:
            concat_data = np.concatenate((data.features, data.target.reshape(-1, 1)), 1)
        else:
            concat_data = np.concatenate((data.features, data.target.reshape(-1, data.target.shape[1])), 1)
        frame = H2OFrame(python_obj=concat_data)
        return frame

    def _get_h2o_connect_config(self):
        ip = '127.0.0.1'
        port = 8888
        return ip, port


class H2OAutoMLClassificationStrategy(EvaluationStrategy):
    __operations_by_types = {
        'h2o_class': H2OAutoML
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        self.model_class = H2OSerializationWrapper
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        ip, port = self._get_h2o_connect_config()

        h2o.init(ip=ip, port=port, name='h2o_server')

        frame = self._data_transform(train_data)

        train_frame, valid_frame = frame.split_frame(ratios=[0.85])

        # make sure that your target column is the last one
        train_columns = train_frame.columns
        target_name = train_columns[-1]
        train_columns.remove(target_name)
        train_frame[target_name] = train_frame[target_name].asfactor()
        model = self.operation_impl(max_models=self.params_for_fit.get("max_models"),
                                    seed=self.params_for_fit.get("seed"),
                                    max_runtime_secs=self.params_for_fit.get("timeout")*60
                                    )

        model.train(x=train_columns, y=target_name, training_frame=train_frame)
        model.leader.classes_ = np.unique(train_data.target)
        return H2OSerializationWrapper([model.leader])

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        frame = self._data_transform(predict_data)
        prediction = trained_operation.get_estimators()[0].predict(frame)
        prediction = prediction.as_data_frame().to_numpy()

        if self.output_mode == 'labels':
            prediction = prediction[:, 0]
        elif self.output_mode in ['probs', 'full_probs', 'default']:
            prediction = prediction[:, 1::]
        else:
            raise ValueError(f'Output model {self.output_mode} is not supported')
        out = self._convert_to_output(prediction, predict_data)
        return out

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain H2O AutoML Classification Strategy for {operation_type}')

    def _data_transform(self, data: InputData) -> H2OFrame:
        concat_data = np.concatenate((data.features, data.target.reshape(-1, 1)), 1)
        frame = H2OFrame(python_obj=concat_data)
        return frame

    def _get_h2o_connect_config(self):
        ip = '127.0.0.1'
        port = 8888
        return ip, port


class TPOTAutoMLRegressionStrategy(EvaluationStrategy):
    __operations_by_types = {
        'tpot_regr': TPOTRegressor
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        if train_data.task.task_type == TaskTypesEnum.ts_forecasting:
            target_len = train_data.task.task_params.forecast_length
        else:
            target_len = 1
        models = []
        if len(train_data.target.shape) == 1:
            target = train_data.target.reshape(-1, 1)
        else:
            target = train_data.target

        for i in range(target.shape[1]):
            model = self.operation_impl(generations=self.params_for_fit.get('generations'),
                                        population_size=self.params_for_fit.get('population_size'),
                                        verbosity=2,
                                        random_state=42,
                                        max_time_mins=self.params_for_fit.get('timeout', 0.) // target_len
                                        )
            model.fit(train_data.features.astype(float), target.astype(float)[:, i])
            models.append(model.fitted_pipeline_)
        model = TPOTRegressionSerializationWrapper(models)
        return model

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        res = []
        features = predict_data.features.astype(float)
        for model in trained_operation.get_estimators():
            prediction = model.predict(features)
            prediction = prediction
            res.append(np.ravel(prediction))
        res = np.hstack(res)
        out = self._convert_to_output(res, predict_data)
        return out

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain H2O AutoML Regression Strategy for {operation_type}')


class TPOTAutoMLClassificationStrategy(EvaluationStrategy):
    __operations_by_types = {
        'tpot_class': TPOTClassifier
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        model = self.operation_impl(generations=self.params_for_fit.get('generations'),
                                    population_size=self.params_for_fit.get('population_size'),
                                    verbosity=2,
                                    random_state=42,
                                    max_time_mins=self.params_for_fit.get('timeout', 0.)
                                    )
        model.classes_ = np.unique(train_data.target)

        model.fit(train_data.features.astype(float), train_data.target.astype(int))

        return model.fitted_pipeline_

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        n_classes = len(trained_operation.classes_)
        if self.output_mode == 'labels':
            prediction = trained_operation.predict(predict_data.features.astype(float))
        elif self.output_mode in ['probs', 'full_probs', 'default']:
            prediction = trained_operation.predict_proba(predict_data.features.astype(float))
            if n_classes < 2:
                raise NotImplementedError()
            elif n_classes == 2 and self.output_mode != 'full_probs' and len(prediction.shape) > 1:
                prediction = prediction[:, 1]
        else:
            raise ValueError(f'Output model {self.output_mode} is not supported')
        out = self._convert_to_output(prediction, predict_data)
        return out

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain H2O AutoML Classification Strategy for {operation_type}')
