from typing import Any, Dict, Type

import numpy as np
from numpy import ndarray

from .. import Serializer


def ndarray_to_json(obj: ndarray) -> Dict[str, Any]:
    return {
        'object': obj.tolist(),
        'dtype': str(obj.dtype),
        **Serializer.dump_path_to_obj(obj)
    }


def ndarray_from_json(cls: Type[ndarray], json_obj: Dict[str, Any]) -> ndarray:
    return np.array(**json_obj)
