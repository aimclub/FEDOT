from typing import Any

from fedot.core.common.registry import Registry
from fedot.core.caching.rules import SaverNotFoundError
from fedot.core.common.registry_predicates import (
    is_tensor_data,
    is_preprocessing_handler,
    is_preprocessing_plan,
)
from fedot.core.caching.inmemory_operations import (
    save_tensor_data,
    save_preprocessing_model,
    save_preprocessing_plan,
)
from fedot.core.caching.responses import SaverResponse


class Saver(Registry):
    not_found_error = SaverNotFoundError
    not_found_message = "No saver function registered for data type: {source_type}"

    @classmethod
    def save(cls, data: Any, hash: str) -> SaverResponse:
        saver_func = cls.resolve_creator(data)
        return saver_func(data, hash)


Saver.register_creator(is_tensor_data)(save_tensor_data)
Saver.register_creator(is_preprocessing_handler)(save_preprocessing_model)
Saver.register_creator(is_preprocessing_plan)(save_preprocessing_plan)
