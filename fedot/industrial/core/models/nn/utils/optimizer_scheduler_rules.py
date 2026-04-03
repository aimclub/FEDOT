from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from inspect import isclass
from typing import Any


@dataclass(frozen=True)
class OptimizerFactoryPlan:
    constructor: Any
    learning_rate: float
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class SchedulerFactoryPlan:
    constructor: Any
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class SchedulerActionPlan:
    renew: bool
    step: bool


def resolve_registry_constructor(spec: Any, registry: Any, *, kind: str) -> Any:
    if isinstance(spec, partial):
        return spec
    if isinstance(spec, str):
        resolved = registry[spec].value
        if isinstance(resolved, tuple):
            return resolved[0]
        return resolved
    if isclass(spec):
        return spec
    raise TypeError(
        f'Unknown type for {kind} is passed! Required: constructor, partial, or str'
    )


def build_optimizer_factory_plan(opt_type: Any,
                                 learning_rate: float,
                                 registry: Any,
                                 **kwargs) -> OptimizerFactoryPlan:
    constructor = resolve_registry_constructor(opt_type, registry, kind='optimizer')
    return OptimizerFactoryPlan(
        constructor=constructor,
        learning_rate=learning_rate,
        kwargs=dict(kwargs),
    )


def instantiate_optimizer_factory(plan: OptimizerFactoryPlan):
    if isinstance(plan.constructor, partial):
        return partial(plan.constructor, lr=plan.learning_rate, **plan.kwargs)
    return partial(plan.constructor, lr=plan.learning_rate, **plan.kwargs)


def build_scheduler_factory_plan(sch_type: Any, registry: Any, **kwargs) -> SchedulerFactoryPlan:
    constructor = resolve_registry_constructor(sch_type, registry, kind='scheduler')
    return SchedulerFactoryPlan(
        constructor=constructor,
        kwargs=dict(kwargs),
    )


def instantiate_scheduler_factory(plan: SchedulerFactoryPlan):
    if isinstance(plan.constructor, partial):
        return partial(plan.constructor, **plan.kwargs)
    return partial(plan.constructor, **plan.kwargs)


def build_scheduler_action_plan(*,
                                epoch: int,
                                scheduler: Any,
                                optimizer: Any,
                                scheduler_step_each: int) -> SchedulerActionPlan:
    if scheduler_step_each <= 0:
        raise ValueError('scheduler_step_each must be positive')

    renew = False
    if scheduler is not None:
        scheduler_optimizer = getattr(scheduler, 'optimizer', None)
        renew = epoch == 1 or scheduler_optimizer is not optimizer

    step = epoch % scheduler_step_each == 0

    return SchedulerActionPlan(
        renew=renew,
        step=step,
    )
