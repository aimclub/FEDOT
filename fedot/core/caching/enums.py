from golem.utilities.data_structures import ComparableEnum as Enum


class CacheModeEnum(Enum):
    """
    Strategy for ``Cacher.clear_cache``.

    Values select whether the full cache tree, explicit tensor hashes, or the
    oldest tensor artifacts are removed.
    """
    ALL = "all"
    TENSOR_DATA = "tensor_data"
    FIRST_N_TENSOR_DATA = "first_n_tensor_data"
