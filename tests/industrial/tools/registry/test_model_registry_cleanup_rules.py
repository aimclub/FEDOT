from fedot.industrial.tools.registry.model_registry_cleanup_rules import (
    build_compressor_cleanup_plan,
    build_dynamic_model_cleanup_plan,
    build_registry_storage_cleanup_plan,
    build_trainer_cleanup_plan,
)


class _DummyModule:
    pass


class _NestedTrainer:
    def __init__(self):
        self.model = _DummyModule()


class _Trainer:
    def __init__(self):
        self.model = _DummyModule()
        self._trainer = _NestedTrainer()


class _Compressor:
    def __init__(self):
        self.model_before = _DummyModule()
        self.model_after = None
        self.trainer = _Trainer()
        self.direct_module = _DummyModule()
        self.module_list = [_DummyModule(), 'plain_value']
        self.module_tuple = (_DummyModule(), 'plain_value')
        self.module_dict = {'first': _DummyModule(), 'second': 'plain_value'}


def test_build_registry_storage_cleanup_plan_detects_checkpoint_bytes_column():
    plan = build_registry_storage_cleanup_plan(['record', 'checkpoint_bytes', 'metrics'])
    assert plan.clear_checkpoint_bytes is True
    assert plan.target_column == 'checkpoint_bytes'


def test_build_registry_storage_cleanup_plan_is_noop_without_column():
    plan = build_registry_storage_cleanup_plan(['record', 'metrics'])
    assert plan.clear_checkpoint_bytes is False
    assert plan.target_column is None


def test_build_trainer_cleanup_plan_detects_direct_and_nested_targets():
    trainer = _Trainer()
    plan = build_trainer_cleanup_plan(trainer)

    assert plan.direct_model_attrs == ('model',)
    assert len(plan.nested_targets) == 1
    assert plan.nested_targets[0].trainer_attr == '_trainer'
    assert plan.nested_targets[0].model_attr == 'model'


def test_build_dynamic_model_cleanup_plan_tracks_direct_sequence_and_mapping_targets():
    compressor = _Compressor()
    plan = build_dynamic_model_cleanup_plan(compressor, module_type=_DummyModule)

    assert 'direct_module' in plan.attr_names
    assert any(target.attr_name == 'module_list' and target.mutable for target in plan.sequence_targets)
    assert any(target.attr_name == 'module_tuple' and not target.mutable for target in plan.sequence_targets)
    assert any(target.attr_name == 'module_dict' and target.keys == ('first',) for target in plan.mapping_targets)


def test_build_compressor_cleanup_plan_collects_static_and_dynamic_cleanup_targets():
    compressor = _Compressor()
    plan = build_compressor_cleanup_plan(
        compressor_object=compressor,
        model_attrs_to_clean=('model_before', 'model_after'),
        module_type=_DummyModule,
    )

    assert plan.model_attrs == ('model_before',)
    assert plan.has_trainer is True
    assert 'direct_module' in plan.dynamic_plan.attr_names