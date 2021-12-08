import base64
from typing import Any, Dict, Type

import numpy as np
from numpy import ndarray

from .. import Serializer


def ndarray_to_json(obj: ndarray) -> Dict[str, Any]:
    return {
        'buffer': base64.b64encode(obj.tobytes()).decode('ascii'),
        'shape': obj.shape,
        'dtype': str(obj.dtype),
        **Serializer.dump_path_to_obj(obj)
    }


def ndarray_from_json(cls: Type[ndarray], json_obj: Dict[str, Any]) -> ndarray:
    json_obj['buffer'] = np.frombuffer(base64.b64decode(json_obj['buffer']))
    return cls(**json_obj)
