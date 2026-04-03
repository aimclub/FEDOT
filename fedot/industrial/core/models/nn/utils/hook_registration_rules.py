from __future__ import annotations

from itertools import chain
from typing import Any, Iterable


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


def build_initialized_hooks(hook_groups: Iterable[Any], params: dict[str, Any], model: Any) -> list[Any]:
    return [hook_class(params, model) for hook_class in iter_enabled_hook_classes(hook_groups, params)]
