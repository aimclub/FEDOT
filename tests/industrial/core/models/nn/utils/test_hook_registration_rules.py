from enum import Enum

from fedot.industrial.core.models.nn.utils.hook_registration_rules import (
    HookRegistrationPlan,
    build_hook_registration_plan,
    build_initialized_hooks,
    instantiate_hook_plan,
    iter_enabled_hook_classes,
    resolve_hook_groups,
)


class _EnabledHook:
    initialized_with = []

    def __init__(self, params, model):
        self.params = params
        self.model = model
        self.initialized_with.append((params, model))

    @classmethod
    def check_init(cls, params):
        return True


class _DisabledHook:
    @classmethod
    def check_init(cls, params):
        return False


class _AnotherEnabledHook:
    def __init__(self, params, model):
        self.params = params
        self.model = model

    @classmethod
    def check_init(cls, params):
        return params.get('allow_second', False)


class _DefaultHookGroup(Enum):
    enabled = _EnabledHook
    disabled = _DisabledHook


class _AdditionalHookGroup(Enum):
    another = _AnotherEnabledHook


def test_resolve_hook_groups_preserves_order_and_removes_duplicates():
    resolved = resolve_hook_groups(
        default_hook_groups=[_DefaultHookGroup, _AdditionalHookGroup],
        additional_hook_groups=[_AdditionalHookGroup, _DefaultHookGroup],
    )

    assert resolved == (_DefaultHookGroup, _AdditionalHookGroup)


def test_iter_enabled_hook_classes_respects_check_init_filters():
    hook_classes = list(
        iter_enabled_hook_classes(
            hook_groups=[_DefaultHookGroup, _AdditionalHookGroup],
            params={'allow_second': True},
        )
    )

    assert hook_classes == [_EnabledHook, _AnotherEnabledHook]


def test_build_initialized_hooks_creates_instances_from_enabled_groups():
    _EnabledHook.initialized_with.clear()
    model = object()
    params = {'allow_second': True}

    hooks = build_initialized_hooks(
        hook_groups=[_DefaultHookGroup, _AdditionalHookGroup],
        params=params,
        model=model,
    )

    assert len(hooks) == 2
    assert isinstance(hooks[0], _EnabledHook)
    assert isinstance(hooks[1], _AnotherEnabledHook)
    assert _EnabledHook.initialized_with == [(params, model)]


def test_build_hook_registration_plan_is_ordered_and_filtered():
    plan = build_hook_registration_plan(
        default_hook_groups=[_DefaultHookGroup],
        additional_hook_groups=[_AdditionalHookGroup],
        params={'allow_second': True},
    )

    assert plan == HookRegistrationPlan(
        hook_groups=(_DefaultHookGroup, _AdditionalHookGroup),
        hook_classes=(_EnabledHook, _AnotherEnabledHook),
    )


def test_instantiate_hook_plan_uses_precomputed_hook_classes():
    _EnabledHook.initialized_with.clear()
    model = object()
    params = {'allow_second': False}
    plan = HookRegistrationPlan(
        hook_groups=(_DefaultHookGroup,),
        hook_classes=(_EnabledHook,),
    )

    hooks = instantiate_hook_plan(plan, params, model)

    assert len(hooks) == 1
    assert isinstance(hooks[0], _EnabledHook)
    assert _EnabledHook.initialized_with == [(params, model)]