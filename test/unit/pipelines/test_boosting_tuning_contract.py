import pytest
from sklearn.datasets import make_classification

from fedot.api.main import Fedot
from fedot.core.operations.evaluation.operation_implementations.models.boostings_implementations import (
    FedotLightGBMClassificationImplementation,
    FedotXGBoostClassificationImplementation,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace


@pytest.mark.parametrize('operation', ['catboost', 'catboostreg'])
def test_catboost_search_space_uses_supported_parameter_names(operation):
    parameters = set(PipelineSearchSpace().get_parameters_for_operation(operation))

    assert {'num_trees', 'max_bin'} <= parameters
    assert parameters.isdisjoint({'iterations', 'border_count', 'max_leaves'})


@pytest.mark.parametrize('operation', ['lgbm', 'lgbmreg'])
def test_lightgbm_search_space_has_no_conflicting_row_wise_parameter(operation):
    parameters = PipelineSearchSpace().get_parameters_for_operation(operation)

    assert not any(parameter.strip() == 'force_row_wise' for parameter in parameters)


def test_lightgbm_early_stopping_callback_is_returned():
    implementation = FedotLightGBMClassificationImplementation(
        OperationParameters(
            early_stopping_rounds=7,
            use_eval_set=True,
            verbose=False,
        )
    )

    callbacks = implementation.update_callbacks()

    assert len(callbacks) == 1
    assert callable(callbacks[0])


@pytest.mark.parametrize(
    ('implementation_class', 'n_jobs_attribute'),
    [
        (FedotLightGBMClassificationImplementation, 'n_jobs'),
        (FedotXGBoostClassificationImplementation, 'n_jobs'),
    ],
)
def test_boosting_implementation_passes_n_jobs_to_library_model(
        implementation_class, n_jobs_attribute):
    implementation = implementation_class(
        OperationParameters(n_jobs=3, use_eval_set=False)
    )

    assert getattr(implementation.model, n_jobs_attribute) == 3


def test_pipeline_n_jobs_replacement_updates_fit_parameters():
    pipeline = Pipeline(PipelineNode('lgbm'))

    pipeline.replace_n_jobs_in_nodes(4)

    assert pipeline.nodes[0].parameters['n_jobs'] == 4
    assert pipeline.nodes[0]._parameters.get('n_jobs') == 4


def test_api_predefined_booster_uses_configured_n_jobs():
    features, target = make_classification(
        n_samples=100,
        n_features=8,
        n_informative=4,
        random_state=42,
    )
    automl = Fedot(problem='classification', n_jobs=3)

    pipeline = automl.fit(features, target, predefined_model='lgbm')

    fitted_model = pipeline.nodes[0].fitted_operation.model
    assert fitted_model.n_jobs == 3
