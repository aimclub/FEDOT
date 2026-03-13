from typing import Dict, Iterable, Optional



def normalize_node_parameters(params: Optional[dict], default_params_stub, nested_params_label: str) -> Dict:
    if params is None:
        return {}
    if params == default_params_stub:
        return {}
    if nested_params_label in params:
        return dict(params[nested_params_label])
    return dict(params)



def merge_node_parameters(current_parameters: Optional[dict], changed_parameters: Optional[dict]) -> Dict:
    return {
        **dict(current_parameters or {}),
        **dict(changed_parameters or {}),
    }



def should_update_node_parameters(operation_type: str, operation_tags: Optional[Iterable[str]]) -> bool:
    if 'atomized' in operation_type:
        return False
    return 'correct_params' in set(operation_tags or ())
