from fedot.api.api_utils.api_composer_rules import build_cache_init_plan, build_tuner_plan


def test_build_cache_init_plan_respects_input_preprocessing_boundary():
    plan = build_cache_init_plan(
        use_operations_cache=True,
        use_preprocessing_cache=True,
        use_predictions_cache=True,
        use_input_preprocessing=False,
        cache_dir='cache',
        use_stats=True,
    )

    assert plan.use_operations_cache is True
    assert plan.use_preprocessing_cache is False
    assert plan.use_predictions_cache is True
    assert plan.cache_dir == 'cache'
    assert plan.use_stats is True


def test_build_tuner_plan_is_deterministic_and_clamps_timeout():
    plan = build_tuner_plan(metrics=['f1', 'roc_auc'], timeout_minutes=-3, iterations=42)

    assert plan.metric == 'f1'
    assert plan.iterations == 42
    assert plan.timeout_minutes == 0.0
