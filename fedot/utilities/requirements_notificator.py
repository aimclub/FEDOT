import warnings


def warn_requirement(name: str, default_install_path: str = '.[extra]', *, should_raise: bool = False):
    """
    :param name: module name failed to load
    :param should_raise: bool indicating if ImportError should be raised
    """
    msg = f'"{name}" is not installed, use "pip install {default_install_path}" to fulfil'
    if should_raise:
        raise ImportError(msg)
    else:
        warnings.warn(f'{msg} or ignore this warning')
