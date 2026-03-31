from fedot.industrial.api.utils.industrial_strategy_rules import (
    build_federated_runtime_plan,
    build_industrial_kernel_finetune_plan,
    build_sampling_iteration_plans,
    build_sampling_predict_plan,
    resolve_industrial_strategy_dispatch,
)
from fedot.industrial.core.repository.constanst_repository import FEDOT_TUNER_STRATEGY, FEDOT_TUNING_METRICS


def test_resolve_industrial_strategy_dispatch_returns_method_names():
    plan = resolve_industrial_strategy_dispatch('federated_automl')

    assert plan.fit_method_name == '_federated_strategy'
    assert plan.predict_method_name == '_federated_predict'


def test_build_federated_runtime_plan_switches_between_raf_and_fedot():
    raf_plan = build_federated_runtime_plan(
        n_samples=10000,
        batch_size_threshold=1000,
        requested_workers=None,
        timeout=10,
        timeout_partition=4,
        default_workers=5,
    )
    fedot_plan = build_federated_runtime_plan(
        n_samples=100,
        batch_size_threshold=1000,
        requested_workers=None,
        timeout=10,
        timeout_partition=4,
        default_workers=5,
    )

    assert raf_plan.use_raf is True
    assert raf_plan.raf_workers == 5
    assert raf_plan.batch_size == 2000
    assert raf_plan.timeout == 2
    assert fedot_plan.use_raf is False
    assert fedot_plan.batch_size is None


def test_build_sampling_predict_plan_tracks_mode_and_cur_feature_space():
    labels_plan = build_sampling_predict_plan(mode='labels', sampling_algorithm='CUR')
    probs_plan = build_sampling_predict_plan(mode='probs', sampling_algorithm='Random')

    assert labels_plan.labels_output is True
    assert labels_plan.use_cur_feature_space is True
    assert probs_plan.labels_output is False
    assert probs_plan.use_cur_feature_space is False


def test_build_industrial_kernel_finetune_plan_normalizes_metric_and_tuner():
    plan = build_industrial_kernel_finetune_plan('classification', {'iterations': 5})

    assert plan.normalized_tuning_params['iterations'] == 5
    assert plan.normalized_tuning_params['metric'] == FEDOT_TUNING_METRICS['classification']
    assert plan.normalized_tuning_params['tuner'] == FEDOT_TUNER_STRATEGY['simultaneous']


def test_build_sampling_iteration_plans_creates_stable_keys():
    plans = build_sampling_iteration_plans('CUR', [0.2, 0.5])

    assert [plan.sampling_rate for plan in plans] == [0.2, 0.5]
    assert [plan.result_key for plan in plans] == ['CUR_sampling_rate_0.2', 'CUR_sampling_rate_0.5']
