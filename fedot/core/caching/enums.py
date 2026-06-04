from golem.utilities.data_structures import ComparableEnum as Enum


class CacheModeEnum(Enum):
    """
    Mode of cache clearing.
    """
    ALL = "all"
    TENSOR_DATA = "tensor_data"
    FIRST_N_TENSOR_DATA = "first_n_tensor_data"
