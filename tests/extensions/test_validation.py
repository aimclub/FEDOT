from dataclasses import replace

import pytest

from fedot.extensions import ModelHyperparamsSchema, validate_extension_manifest
from fedot.extensions.execution_rules import plan_execution
from fedot.extensions.parameter_rules import resolve_extension_params
from fedot.extensions.registry import smoke_test_extension
from tests.extensions.test_registration_scope import manifest, transform_manifest


@pytest.mark.parametrize('field,value,code', [
    ('name', None, 'empty_extension_name'),
    ('version', 42, 'empty_extension_version'),
    ('models', None, 'invalid_manifest_operations'),
    ('models', (42,), 'invalid_model_spec_type'),
])
def test_malformed_manifest_has_typed_failure(field, value, code):
    result = validate_extension_manifest(replace(manifest(), **{field: value}))
    assert result.monoid[0].code == code


@pytest.mark.parametrize('change,code', [
    ({'name': None}, 'empty_model_name'),
    ({'name': 'model/suffix'}, 'invalid_operation_name'),
    ({'capabilities': None}, 'invalid_capabilities'),
    ({'factory': None}, 'invalid_model_factory'),
    ({'hyperparams_schema': None}, 'invalid_hyperparams_schema'),
])
def test_malformed_spec_has_typed_failure(change, code):
    item = manifest()
    item = replace(item, models=(replace(item.models[0], **change),))
    assert validate_extension_manifest(item).monoid[0].code == code


@pytest.mark.parametrize('field,value', [('tasks', None), ('data_types', ('table',)),
                                         ('tags', (None,)), ('backend', 'numpy'),
                                         ('requires_target', 'yes')])
def test_capabilities_require_typed_values(field, value):
    item = manifest()
    spec = item.models[0]
    spec = replace(spec, capabilities=replace(spec.capabilities, **{field: value}))
    assert validate_extension_manifest(replace(item, models=(spec,))).monoid[0].code == 'invalid_capabilities'


@pytest.mark.parametrize('item,spec_field', [
    (manifest(), 'models'),
    (transform_manifest(), 'transforms'),
])
def test_reserved_behavior_tags_are_rejected_for_every_operation_kind(item, spec_field):
    spec = getattr(item, spec_field)[0]
    spec = replace(spec, capabilities=replace(spec.capabilities, tags=('correct_params',)))

    result = validate_extension_manifest(replace(item, **{spec_field: (spec,)}))

    assert result.is_left()
    assert result.monoid[0].code == 'reserved_behavior_tag'
    assert result.monoid[0].details == {
        'operation': spec.name,
        'tags': ['correct_params'],
    }


def test_non_default_remains_a_valid_catalog_tag():
    item = manifest()
    spec = item.models[0]
    spec = replace(spec, capabilities=replace(spec.capabilities, tags=('non-default',)))

    assert validate_extension_manifest(replace(item, models=(spec,))).is_right()


def test_model_and_transform_capabilities_are_not_interchangeable():
    model = manifest().models[0]
    transform = transform_manifest().transforms[0]
    result = resolve_extension_params(replace(model, capabilities=transform.capabilities))
    assert result.monoid[0].code == 'invalid_capabilities'


@pytest.mark.parametrize('params', [False, 42, [], {1: 'value'}])
def test_invalid_parameter_mapping_never_reaches_factory(params):
    calls = []
    spec = manifest(factory=lambda: calls.append('factory')).models[0]
    assert resolve_extension_params(spec, params).monoid[0].code == 'invalid_parameters'
    assert calls == []


def test_explicit_none_default_is_not_missing():
    spec = replace(manifest().models[0], hyperparams_schema=ModelHyperparamsSchema(
        required=('optional_value',), defaults={'optional_value': None}))
    assert resolve_extension_params(spec).value == {'optional_value': None}


@pytest.mark.parametrize('schema', [ModelHyperparamsSchema(required=('x',), optional=('x',)),
                                    ModelHyperparamsSchema(defaults={'_hidden': 1}),
                                    ModelHyperparamsSchema(optional=('model_fit',))])
def test_invalid_parameter_schema_is_rejected(schema):
    result = resolve_extension_params(replace(manifest().models[0], hyperparams_schema=schema))
    assert result.monoid[0].code == 'invalid_hyperparams_schema'


def test_smoke_validates_all_parameters_before_any_factory():
    calls = []
    item = manifest(factory=lambda: calls.append('factory'))
    second = replace(item.models[0], name='second', hyperparams_schema=ModelHyperparamsSchema(required=('x',)))
    result = smoke_test_extension(replace(item, models=item.models + (second,)))
    assert result.monoid[0].code == 'missing_required_hyperparams'
    assert calls == []


def test_execution_plan_has_no_runtime_effects():
    calls = []
    spec = manifest(factory=lambda: calls.append('factory')).models[0]
    caps = spec.capabilities
    result = plan_execution(spec, caps.tasks[0], caps.data_types[0], fitting=True, has_target=True)
    assert result.is_right()
    assert result.value.operation_name == spec.name
    assert calls == []


@pytest.mark.parametrize('parameters', [42, False, {'missing': {}}, {'test_model': []}])
def test_invalid_smoke_parameter_mapping_has_typed_failure(parameters):
    calls = []
    result = smoke_test_extension(manifest(factory=lambda: calls.append('factory')), parameters)
    assert result.monoid[0].code == 'invalid_parameters'
    assert calls == []


@pytest.mark.parametrize('data_type', ['unknown', [], None])
def test_execution_plan_rejects_malformed_data_type(data_type):
    spec = manifest().models[0]
    result = plan_execution(spec, spec.capabilities.tasks[0], data_type)
    assert result.monoid[0].code == 'unsupported_data_type'
