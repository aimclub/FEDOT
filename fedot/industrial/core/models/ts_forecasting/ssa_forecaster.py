from copy import deepcopy
from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from golem.core.tuning.simultaneous import SimultaneousTuner
from pymonad.either import Either

from fedot.industrial.core.architecture.preprocessing.data_convertor import FedotConverter
from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.operation.transformation.basis.eigen_basis import EigenBasisImplementation
from fedot.industrial.core.operation.transformation.data.hankel import HankelMatrix
from fedot.industrial.core.operation.transformation.regularization.spectrum import reconstruct_basis
from fedot.industrial.core.repository.constanst_repository import FEDOT_TUNING_METRICS
from fedot.industrial.core.repository.industrial_implementations.abstract import build_tuner


class SSAForecasterImplementation(ModelImplementation):
    """Model for forecasting univariate timeseries with Singular Spectrum Decomposition.
    For given time series ``T`` we construct trajectory matrix (hankel matrix) ``X``, where
    ``X = U x S x V_t``. After decomposition, we forecast ``V_t`` rows separately, and after
    that reconstruct basis. Note that we use only few components to reconstruct basis. Other
    components considered as error (we just sample them).

    Attributes:
        window_size_method: str, method for estimating window size for SSA forecaster

    Example:
        To use this operation you can create pipeline as follows::

            from fedot.industrial.core.architecture.settings.computational import backend_methods as np
            from fedot.core.data.data import InputData
            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from examples.example_utils import get_ts_data
            from fedot.industrial.core.repository.initializer_industrial_models import IndustrialModels

            forecast_length = 13
            train_data, test_data, dataset_name = get_ts_data('m4_monthly', forecast_length)
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('ssa_forecaster').build()
                pipeline.fit(train_data)
                prediction = pipeline.predict(test_data)
                print(prediction)

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        tuning_params = {'tuning_iterations': 100,
                         'tuning_timeout': 20,
                         'tuning_early_stop': 20,
                         'tuner': SimultaneousTuner}
        component_mode_dict = {
            'topological': PipelineBuilder().add_node('lagged').add_node('topological_features').add_node('treg'),
            'ar': PipelineBuilder().add_node('ar')
        }

        self.window_size_method = params.get('window_size_method')
        self.history_lookback = max(params.get('history_lookback', 0), 100)
        self.low_rank_approximation = params.get(
            'low_rank_approximation', False)
        self.tuning_params = params.get('tuning_params', tuning_params)
        self.component_model = params.get('component_model', 'topological')
        self.mode = params.get('mode', 'channel_independent')
        self.component_model = component_mode_dict[self.component_model]
        self.trend_model = PipelineBuilder().add_node('lagged').add_node('ridge')
        self.trend_model = self.component_model
        self._decomposer = None
        self._rank_thr = None
        self._window_size = None
        self.horizon = None
        self.preprocess_to_lagged = False

    def _tune_component_model(self, model_to_tune, component):
        self.tuning_params['metric'] = FEDOT_TUNING_METRICS['regression']
        pipeline_tuner, component_model = build_tuner(self,
                                                      model_to_tune,
                                                      self.tuning_params,
                                                      component,
                                                      'head')
        component_model.fit(component)
        reconstructed_forecast = component_model.predict(
            component).predict[-self.horizon:]
        return reconstructed_forecast, component_model

    def _combine_trajectory(self, U, VT, n_components):
        if len(self._rank_thr) > 2:
            self.PCT = np.concatenate([U[:, 0].reshape(
                1, - 1), np.array([np.sum([U[:, i], U[:, i + 1]], axis=0) for i in self._rank_thr if i != 0 and i % 2 != 0])]).T

            current_dynamics = np.concatenate([VT[0, :].reshape(1, -1), np.array([np.sum(
                [VT[i, :], VT[i + 1, :]], axis=0) for i in self._rank_thr if i != 0 and i % 2 != 0])])
        else:
            self.PCT, current_dynamics = U[:,
                                           :n_components], VT[:n_components, :]

        return current_dynamics

    def predict(self, input_data: InputData) -> OutputData:
        hankel_matrix = HankelMatrix(
            time_series=input_data.features,
            window_size=self._decomposer.window_size).trajectory_matrix
        U, s, VT = np.linalg.svd(hankel_matrix)
        n_components = max(2, len(self._rank_thr))
        current_dynamics = self._combine_trajectory(U, VT, n_components)

        if self.mode == 'one_dimensional':
            comp = deepcopy(input_data)
            basis = reconstruct_basis(U=self.PCT,
                                      Sigma=s[:self.PCT.shape[1]],
                                      VT=current_dynamics,
                                      ts_length=input_data.features.shape[0])
            comp.features, comp.idx = np.array(basis).sum(
                axis=1), np.arange(np.array(basis).sum(axis=1).shape[0])

            reconstructed_forecast, self.model_by_channel = self._tune_component_model(
                self.trend_model.build(), comp)
        elif self.mode == 'channel_independent':
            forecast_by_channel, self.model_by_channel = self._predict_channel(
                input_data, current_dynamics, self.horizon)

            self.forecasted_dynamics = np.concatenate(
                [current_dynamics, np.vstack(list(forecast_by_channel.values()))], axis=1)
            basis = reconstruct_basis(U=self.PCT,
                                      Sigma=s[:self.PCT.shape[1]],
                                      VT=self.forecasted_dynamics,
                                      ts_length=input_data.features.shape[0] + self.horizon)

            summed_basis = np.array(basis).sum(axis=1)
            reconstructed_forecast = summed_basis[-self.horizon:]

        prediction = reconstructed_forecast
        predict_data = FedotConverter(input_data).convert_to_output_data(
            prediction=prediction,
            predict_data=input_data,
            output_data_type=input_data.data_type)
        return predict_data

    def fit(self, input_data: InputData):
        pass

    def __predict_for_fit(self, ts):
        basis = self._decomposer.transform(ts)
        self._rank_thr = list(range(basis.predict.shape[0] - 1))
        if len(self._rank_thr) > 2:
            components_correlation = np.concatenate([basis.predict[0, :].reshape(1, -1), np.array([np.sum(
                [basis.predict[i, :], basis.predict[i + 1, :]], axis=0) for i in self._rank_thr
                if i != 0 and i % 2 != 0])])
        else:
            components_correlation = basis.predict

        reconstructed_features = np.array(components_correlation).sum(axis=0)
        return reconstructed_features

    def predict_for_fit(self, input_data: InputData) -> np.ndarray:
        if self.horizon is None:
            self.horizon = input_data.task.task_params.forecast_length
        if input_data.features.shape[0] > self.history_lookback:
            self.history_lookback = round(input_data.features.shape[0] * 0.2) \
                if self.history_lookback == 0 else self.history_lookback
            input_data.features = input_data.features[-self.history_lookback:].squeeze(
            )
        else:
            self.history_lookback = None
        self._decomposer = EigenBasisImplementation(
            OperationParameters(
                low_rank_approximation=self.low_rank_approximation,
                rank_regularization='explained_dispersion'
            )
        )

        predict = self.__predict_for_fit(input_data)
        return predict

    def _predict_channel(
            self,
            input_data: InputData,
            component_dynamics,
            forecast_length: int):
        comp = deepcopy(input_data)
        comp.features, comp.target, comp.idx = component_dynamics, component_dynamics, \
            np.arange(component_dynamics.shape[1])
        forecast_by_channel, model_by_channel = {}, {}
        for index, ts_comp in enumerate(comp.features):
            comp.features = ts_comp
            comp.target = ts_comp
            model_to_tune = self.component_model.build(
            ) if index != 0 else self.trend_model.build()
            reconstructed_forecast, component_model = self._tune_component_model(
                model_to_tune, comp)
            forecast_by_channel.update(
                {f'{index}_channel': reconstructed_forecast})
            model_by_channel.update({f'{index}_channel': component_model})

        return forecast_by_channel, model_by_channel

    def reconstruct_basis(self, U, s, VT):
        return Either.insert([U, s, VT]).then(
            self._decomposer.data_driven_basis).value[0]
