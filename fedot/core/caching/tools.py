from fedot.core.utils import CACHE_DIR


def ensure_cache_dirs() -> None:
    for subdir in (
        "tensor_data",
        "preprocessing_models",
        "preprocessing_plans",
        "traces",
    ):
        (CACHE_DIR / subdir).mkdir(parents=True, exist_ok=True)
