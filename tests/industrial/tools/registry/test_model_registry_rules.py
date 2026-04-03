from fedot.industrial.tools.registry.model_registry_rules import (
    RegistryRecordPlan,
    build_registry_record_plan,
    build_registry_record_update_plan,
    build_registry_stage_mode_plan,
    merge_registry_metrics,
    normalize_registry_stage,
)


class _DummyTrainer:
    pass


def test_normalize_registry_stage_handles_before_and_after_aliases():
    assert normalize_registry_stage(None) is None
    assert normalize_registry_stage('before_fit') == 'before'
    assert normalize_registry_stage('initial_checkpoint') == 'before'
    assert normalize_registry_stage('after_eval') == 'after'
    assert normalize_registry_stage('unknown_stage') == 'after'


def test_build_registry_stage_mode_plan_prefers_explicit_mode():
    plan = build_registry_stage_mode_plan(
        stage='before_fit',
        mode='gpu_bridge',
        latest_record={'mode': 'inherited_mode'},
        trainer=_DummyTrainer(),
    )

    assert plan.stage == 'before'
    assert plan.mode == 'gpu_bridge'
    assert plan.mode_source == 'explicit'


def test_build_registry_stage_mode_plan_inherits_latest_record_mode():
    plan = build_registry_stage_mode_plan(
        stage='after_train',
        mode=None,
        latest_record={'mode': 'baseline_gpu'},
    )

    assert plan.stage == 'after'
    assert plan.mode == 'baseline_gpu'
    assert plan.mode_source == 'inherited'


def test_build_registry_stage_mode_plan_uses_trainer_name_as_last_resort():
    plan = build_registry_stage_mode_plan(
        stage='after_train',
        mode=None,
        latest_record=None,
        trainer=_DummyTrainer(),
    )

    assert plan.stage == 'after'
    assert plan.mode == '_DummyTrainer'
    assert plan.mode_source == 'trainer'


def test_build_registry_record_plan_returns_typed_empty_metrics_record():
    plan = build_registry_record_plan(
        record_id='record_1',
        fedcore_id='fedcore_1',
        model_id='model_1',
        version='2026-04-03T12:00:00',
        checkpoint_path='checkpoints/model_1.pt',
        model_path='models/model_1.onnx',
        stage='before',
        mode='tensor_gpu_bridge',
    )

    assert isinstance(plan, RegistryRecordPlan)
    assert plan.record == 'record_1'
    assert plan.fedcore == 'fedcore_1'
    assert plan.model == 'model_1'
    assert plan.checkpoint_path == 'checkpoints/model_1.pt'
    assert plan.model_path == 'models/model_1.onnx'
    assert plan.stage == 'before'
    assert plan.mode == 'tensor_gpu_bridge'
    assert plan.metrics == {}


def test_merge_registry_metrics_merges_dicts_and_preserves_non_dict_payloads():
    assert merge_registry_metrics({'loss': 0.2}, {'accuracy': 0.9}) == {'loss': 0.2, 'accuracy': 0.9}
    assert merge_registry_metrics('old', {'accuracy': 0.9}) == {'accuracy': 0.9}
    assert merge_registry_metrics({'loss': 0.2}, 'raw_metric_blob') == 'raw_metric_blob'


def test_build_registry_record_update_plan_normalizes_stage_and_mode():
    plan = build_registry_record_update_plan(
        current_metrics={'loss': 0.4},
        new_metrics={'accuracy': 0.95},
        stage='initial_fit',
        mode=None,
        trainer=_DummyTrainer(),
    )

    assert plan.stage == 'before'
    assert plan.mode == '_DummyTrainer'
    assert plan.metrics == {'loss': 0.4, 'accuracy': 0.95}
