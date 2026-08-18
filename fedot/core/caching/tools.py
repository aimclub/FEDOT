from fedot.core.utils import CACHE_DIR


def ensure_cache_dirs() -> None:
    """
    Create the standard FEDOT on-disk cache directory layout under ``CACHE_DIR``.

    Ensures ``tensor_data``, ``preprocessing_models``, ``preprocessing_plans``,
    ``operations``, and ``traces`` subdirectories exist before read/write operations.
    """
    for subdir in (
        "tensor_data",
        "preprocessing_models",
        "preprocessing_plans",
        "operations",
        "traces",
    ):
        (CACHE_DIR / subdir).mkdir(parents=True, exist_ok=True)
