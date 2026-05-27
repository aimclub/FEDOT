from fedot.core.utils import cache_dir

CACHE_DIR = cache_dir()

def ensure_cache_dirs() -> None:
    for subdir in (
        "arrays",
        "tensor_data",
        "preprocessing_models",
        "plans",
    ):
        (CACHE_DIR / subdir).mkdir(parents=True, exist_ok=True)
