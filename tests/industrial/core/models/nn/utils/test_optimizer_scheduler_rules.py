from functools import partial

from fedot.industrial.core.models.nn.utils.optimizer_scheduler_rules import (
    OptimizerFactoryPlan,
    SchedulerActionPlan,
    SchedulerFactoryPlan,
    build_optimizer_factory_plan,
    build_scheduler_action_plan,
    build_scheduler_factory_plan,
    instantiate_optimizer_factory,
    instantiate_scheduler_factory,
    resolve_registry_constructor,
)


class _RegistryItem:
    def __init__(self, value):
        self.value = value


class _Registry(dict):
    pass


class _OptimizerCtor:
    def __init__(self, params, lr=None, momentum=None):
        self.params = params
        self.lr = lr
        self.momentum = momentum


class _SchedulerCtor:
    def __init__(self, optimizer, learning_rate, epochs):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.step_calls = 0

    def step(self):
        self.step_calls += 1


def test_resolve_registry_constructor_supports_string_and_tuple_registry_entries():
    registry = _Registry(
        adam=_RegistryItem(_OptimizerCtor),
        one_cycle=_RegistryItem((_SchedulerCtor, {'unused': 'mapping'})),
    )

    assert resolve_registry_constructor('adam', registry, kind='optimizer') is _OptimizerCtor
    assert resolve_registry_constructor('one_cycle', registry, kind='scheduler') is _SchedulerCtor


def test_build_optimizer_factory_plan_captures_constructor_and_learning_rate():
    registry = _Registry(adam=_RegistryItem(_OptimizerCtor))

    plan = build_optimizer_factory_plan('adam', 0.05, registry, momentum=0.9)

    assert plan == OptimizerFactoryPlan(
        constructor=_OptimizerCtor,
        learning_rate=0.05,
        kwargs={'momentum': 0.9},
    )


def test_instantiate_optimizer_factory_returns_callable_with_bound_learning_rate():
    plan = OptimizerFactoryPlan(
        constructor=_OptimizerCtor,
        learning_rate=0.15,
        kwargs={'momentum': 0.2},
    )

    factory = instantiate_optimizer_factory(plan)
    optimizer = factory(['p1'])

    assert isinstance(optimizer, _OptimizerCtor)
    assert optimizer.params == ['p1']
    assert optimizer.lr == 0.15
    assert optimizer.momentum == 0.2


def test_build_scheduler_factory_plan_uses_registry_constructor():
    registry = _Registry(one_cycle=_RegistryItem((_SchedulerCtor, {'unused': 'mapping'})))

    plan = build_scheduler_factory_plan('one_cycle', registry)

    assert plan == SchedulerFactoryPlan(
        constructor=_SchedulerCtor,
        kwargs={},
    )


def test_instantiate_scheduler_factory_returns_callable_for_runtime_arguments():
    plan = SchedulerFactoryPlan(
        constructor=_SchedulerCtor,
        kwargs={},
    )

    factory = instantiate_scheduler_factory(plan)
    scheduler = factory('optimizer', 0.01, 5)

    assert isinstance(scheduler, _SchedulerCtor)
    assert scheduler.optimizer == 'optimizer'
    assert scheduler.learning_rate == 0.01
    assert scheduler.epochs == 5


def test_build_scheduler_action_plan_marks_renew_on_first_epoch_and_step_on_interval():
    scheduler = _SchedulerCtor('optimizer-a', 0.1, 5)

    plan = build_scheduler_action_plan(
        epoch=1,
        scheduler=scheduler,
        optimizer='optimizer-a',
        scheduler_step_each=1,
    )

    assert plan == SchedulerActionPlan(renew=True, step=True)


def test_build_scheduler_action_plan_marks_renew_when_optimizer_changes():
    scheduler = _SchedulerCtor('optimizer-a', 0.1, 5)

    plan = build_scheduler_action_plan(
        epoch=2,
        scheduler=scheduler,
        optimizer='optimizer-b',
        scheduler_step_each=3,
    )

    assert plan == SchedulerActionPlan(renew=True, step=False)


def test_build_scheduler_action_plan_rejects_non_positive_step_interval():
    try:
        build_scheduler_action_plan(
            epoch=2,
            scheduler=_SchedulerCtor('optimizer-a', 0.1, 5),
            optimizer='optimizer-a',
            scheduler_step_each=0,
        )
    except ValueError as exc:
        assert 'scheduler_step_each must be positive' in str(exc)
    else:
        raise AssertionError('Expected ValueError for non-positive scheduler_step_each')


def test_resolve_registry_constructor_supports_partial_directly():
    direct = partial(_OptimizerCtor, momentum=0.4)

    resolved = resolve_registry_constructor(direct, _Registry(), kind='optimizer')

    assert resolved is direct