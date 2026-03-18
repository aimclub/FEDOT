import torch


class Backend:
    xp = None
    pd = None
    device = None
    name = None
    is_initialized = False


def set_backend(name: str = "cpu"):
    if name == "gpu":
        import cupy as xp
        import cudf as pd

        Backend.xp = xp
        Backend.pd = pd
        Backend.device = torch.device("cuda")
        Backend.name = "gpu"

    else:
        import numpy as xp
        import pandas as pd

        Backend.xp = xp
        Backend.pd = pd
        Backend.device = torch.device("cpu")
        Backend.name = "cpu"
    
    Backend.is_initialized = True


def ensure_backend(name: str = "cpu"):
    if not Backend.is_initialized:
        set_backend(name)


def to_xp_array(array):
    xp = Backend.xp
    if xp.__name__ == "numpy":
        if hasattr(array, "get"):
            array = array.get()
        array = xp.array(array)

    elif xp.__name__ == "cupy":
        array = xp.array(array)