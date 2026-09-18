from typing import Dict, Iterable, Tuple


def merge_operation_default_params(default_parameters: Dict, passed_parameters: Dict) -> Dict:
    return {
        **dict(default_parameters or {}),
        **dict(passed_parameters or {}),
    }


def collect_changed_keys(current_parameters: Dict, updated_parameters: Dict,
                         existing_changed_keys: Iterable[str]) -> Tuple[str, ...]:
    changed_keys = list(existing_changed_keys)
    for key, value in updated_parameters.items():
        if key not in changed_keys and current_parameters.get(key) != value:
            changed_keys.append(key)
    return tuple(changed_keys)


def resolve_setdefault_value(current_parameters: Dict, key, value):
    if key in current_parameters:
        return current_parameters[key], False
    return value, True
