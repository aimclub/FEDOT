from typing import Any

from fedot.core.common.registry import Registry
from fedot.core.caching.rules import SaverNotFoundError
from fedot.core.common.registry_predicates import is_tensor_data, is_preprocessing_handler
from fedot.core.caching.inmemory_operations import (
    save_tensor_data, save_preprocessing_model
)
from fedot.core.caching.responses import SaverResponse
from fedot.core.data.tensor_data import TensorData


class Saver(Registry):
    not_found_error = SaverNotFoundError
    not_found_message = "No saver function registered for data type: {source_type}"

    @classmethod
    def save(cls, data: Any, hash: str) -> SaverResponse:
        saver_func = cls.resolve_creator(data)
        return saver_func(data, hash)


@Saver.register_creator(is_tensor_data)
def save_tensor_data_registered(data: TensorData, hash: str) -> SaverResponse:
    return save_tensor_data(data, hash)


@Saver.register_creator(is_preprocessing_handler)
def save_preprocessing_model_registered(
    data: Any,
    hash: str,
) -> SaverResponse:
    return save_preprocessing_model(data, hash)
