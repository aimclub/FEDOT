from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Any, Iterable


@dataclass(frozen=True)
class HookRegistrationPlan:
    hook_groups: tuple[Any, ...]
    hook_classes: tuple[type, ...]


def resolve_hook_groups(default_hook_groups: Iterable[Any],
                        additional_hook_groups: Iterable[Any] | None = None) -> tuple[Any, ...]:
    ordered_groups = []
    seen = set()

    for hook_group in chain(default_hook_groups or (), additional_hook_groups or ()):
        if hook_group is None:
            continue
        marker = id(hook_group)
        if marker in seen:
            continue
        seen.add(marker)
        ordered_groups.append(hook_group)

    return tuple(ordered_groups)


def iter_enabled_hook_classes(hook_groups: Iterable[Any], params: dict[str, Any]) -> Iterable[type]:
    for hook_group in hook_groups:
        for hook_elem in hook_group:
            hook_class = hook_elem.value
            if hook_class.check_init(params):
                yield hook_class


def build_hook_registration_plan(default_hook_groups: Iterable[Any],
                                 additional_hook_groups: Iterable[Any] | None,
                                 params: dict[str, Any]) -> HookRegistrationPlan:
    hook_groups = resolve_hook_groups(default_hook_groups, additional_hook_groups)
    hook_classes = tuple(iter_enabled_hook_classes(hook_groups, params))
    return HookRegistrationPlan(
        hook_groups=hook_groups,
        hook_classes=hook_classes,
    )


def instantiate_hook_plan(plan: HookRegistrationPlan, params: dict[str, Any], model: Any) -> list[Any]:
    return [hook_class(params, model) for hook_class in plan.hook_classes]


def build_initialized_hooks(hook_groups: Iterable[Any], params: dict[str, Any], model: Any) -> list[Any]:
    plan = build_hook_registration_plan(hook_groups, None, params)
    return instantiate_hook_plan(plan, params, model)
