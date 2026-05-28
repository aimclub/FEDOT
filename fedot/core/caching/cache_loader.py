from typing import Any

from fedot.core.common.registry import Registry
from fedot.core.caching.inmemory_operations import load_pkl_file, load_pt_file
from fedot.core.caching.rules import LoaderNotFoundError
from fedot.core.common.registry_predicates import is_pt_filepath, is_pkl_filepath


class Loader(Registry):
    not_found_error = LoaderNotFoundError
    not_found_message = "No loader function registered for data type: {source_type}"

    @classmethod
    def load(cls, source: Any, hash: str=None, kind: str=None) -> Any:
        loader_func = cls.resolve_creator(source)
        return loader_func(source, hash, kind)


@Loader.register_creator(is_pt_filepath)
def load_pt_filepath(source: str, hash: str, kind: str) -> Any:
    return load_pt_file(source, hash, kind)


@Loader.register_creator(is_pkl_filepath)
def load_pkl_filepath(source: str, hash: str, kind: str) -> Any:
    return load_pkl_file(source, hash, kind)