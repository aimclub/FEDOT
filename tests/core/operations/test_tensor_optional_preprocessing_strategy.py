import numpy as np
import pytest
import torch

from fedot.core.data.common.enums import StateEnum
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
from fedot.core.operations.evaluation.optional_preprocessing import (
    TensorOptionalPreprocessingStrategy,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.industrial.core.architecture.preprocessing.ts_optional_service import (
    OptionalTSService,
)
from fedot.preprocessing.service.optional_service import OptionalService
from fedot.preprocessing.service.tabular_optional_service import OptionalTabularService
from fedot.preprocessing.tools.preprocessor_types import (
    ImputationMethodEnum,
    PreprocessingStepEnum,
    ScalingMethodEnum,
)
from fedot.validation.errors import FedotValidationError


def _train_features_with_nan() -> np.ndarray:
    return np.array(
        [
            [1.0, 2.0, 0.0],
            [4.0, np.nan, 1.0],
            [7.0, 8.0, 0.0],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def train_td() -> TensorData:
    return TensorDataCreator.create(_train_features_with_nan(), backend_name='cpu')


@pytest.fixture
def test_td() -> TensorData:
    # Keep FIT state: obligatory predict path requires a train trace_uuid.
    features = np.array(
        [
            [10.0, np.nan, 0.0],
            [11.0, 12.0, 1.0],
        ],
        dtype=np.float32,
    )
    return TensorDataCreator.create(features, backend_name='cpu')


@pytest.mark.unit
def test_fit_returns_fitted_tabular_service_with_default_strategy(train_td):
    strategy = TensorOptionalPreprocessingStrategy(
        'optional_preprocessing',
        OperationParameters(),
    )

    fitted = strategy.fit(train_td)

    assert isinstance(fitted, OptionalTabularService)
    assert fitted.plan is not None
    assert fitted.fitted_handlers is not None
    assert len(fitted.plan.steps) == len(fitted.fitted_handlers)
    assert len(fitted.fitted_handlers) > 0


@pytest.mark.unit
def test_fit_selects_ts_service_for_time_series_data():
    train_td = TensorDataCreator.create(
        np.array(
            [
                [1.0, 10.0, 0.0],
                [2.0, np.nan, 1.0],
                [3.0, 30.0, 0.0],
            ],
            dtype=np.float32,
        ),
        backend_name='cpu',
        data_type='time_series',
    )
    strategy = TensorOptionalPreprocessingStrategy(
        'optional_preprocessing',
        OperationParameters(strategy={PreprocessingStepEnum.imputation: None}),
    )

    fitted = strategy.fit(train_td)

    assert isinstance(fitted, OptionalTSService)


@pytest.mark.unit
def test_fit_uses_explicit_strategy(train_td):
    strategy = TensorOptionalPreprocessingStrategy(
        'optional_preprocessing',
        OperationParameters(
            strategy={
                PreprocessingStepEnum.imputation: [{
                    'method': ImputationMethodEnum.mean,
                    'features_idx': [1],
                    'step_args': None,
                }]
            },
            auto=False,
        ),
    )

    fitted = strategy.fit(train_td)

    assert len(fitted.plan.steps) == 1
    assert fitted.plan.steps[0].step == PreprocessingStepEnum.imputation
    assert fitted.plan.steps[0].method == ImputationMethodEnum.mean


@pytest.mark.unit
def test_fit_with_auto_false_and_no_strategy_builds_empty_plan(train_td):
    strategy = TensorOptionalPreprocessingStrategy(
        'optional_preprocessing',
        OperationParameters(auto=False),
    )

    fitted = strategy.fit(train_td)

    assert isinstance(fitted, OptionalService)
    assert fitted.plan is not None
    assert fitted.fitted_handlers == []
    assert len(fitted.plan.steps) == 0


@pytest.mark.unit
def test_fit_passes_use_cache_to_service(train_td):
    strategy = TensorOptionalPreprocessingStrategy(
        'optional_preprocessing',
        OperationParameters(
            use_cache=False,
            strategy={PreprocessingStepEnum.imputation: None},
        ),
    )

    fitted = strategy.fit(train_td)

    assert fitted.use_cache is False


@pytest.mark.unit
def test_predict_applies_fitted_handlers_without_refit(train_td, test_td):
    strategy = TensorOptionalPreprocessingStrategy(
        'optional_preprocessing',
        OperationParameters(
            strategy={PreprocessingStepEnum.imputation: None},
        ),
    )
    fitted = strategy.fit(train_td)
    handlers_before = list(fitted.fitted_handlers)

    predicted = strategy.predict(fitted, test_td)

    assert isinstance(predicted, TensorData)
    assert predicted.features.shape == test_td.features.shape
    assert not torch.isnan(predicted.features[:, 1]).any()
    assert predicted.features[0, 1] == 5.0
    assert fitted.fitted_handlers == handlers_before


@pytest.mark.unit
def test_predict_for_fit_matches_predict(train_td):
    strategy = TensorOptionalPreprocessingStrategy(
        'optional_preprocessing',
        OperationParameters(
            strategy={PreprocessingStepEnum.imputation: None},
        ),
    )
    fitted = strategy.fit(train_td)

    via_predict = strategy.predict(fitted, train_td)
    via_predict_for_fit = strategy.predict_for_fit(fitted, train_td)

    assert torch.allclose(via_predict.features, via_predict_for_fit.features, equal_nan=True)
    assert not torch.isnan(via_predict_for_fit.features[:, 1]).any()
    assert via_predict_for_fit.features[1, 1] == 5.0


@pytest.mark.unit
def test_fit_raises_for_unsupported_data_type():
    unsupported = TensorData(
        state=StateEnum.FIT,
        features=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.image,
        idx=np.array([0, 1]),
        target=np.array([0, 1]),
    )
    strategy = TensorOptionalPreprocessingStrategy('optional_preprocessing')

    with pytest.raises(FedotValidationError, match='Optional preprocessing is not supported'):
        strategy.fit(unsupported)


@pytest.mark.unit
def test_fit_uses_flat_imputation_and_scaling_params(train_td):
    strategy = TensorOptionalPreprocessingStrategy(
        'optional_preprocessing',
        OperationParameters(
            use_imputation=True,
            imputation_method='mean',
            use_scaling=False,
            scaling_method='none',
        ),
    )

    fitted = strategy.fit(train_td)

    assert len(fitted.plan.steps) == 1
    assert fitted.plan.steps[0].step == PreprocessingStepEnum.imputation
    assert fitted.plan.steps[0].method == ImputationMethodEnum.mean


@pytest.mark.unit
def test_build_optional_strategy_from_flat_params(train_td):
    from fedot.core.operations.evaluation.optional_preprocessing_strategy_builder import (
        build_optional_strategy_from_node_params,
    )

    strategy = build_optional_strategy_from_node_params(
        train_td,
        {
            'use_imputation': True,
            'imputation_method': 'median',
            'use_scaling': True,
            'scaling_method': 'standard',
        },
    )

    assert PreprocessingStepEnum.imputation in strategy
    assert strategy[PreprocessingStepEnum.imputation] == ImputationMethodEnum.median
    assert PreprocessingStepEnum.scaling in strategy
    assert strategy[PreprocessingStepEnum.scaling] == ScalingMethodEnum.standard
