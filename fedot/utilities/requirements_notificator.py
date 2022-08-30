from fedot.core.log import default_log


def warn_requirement(name: str, default_install_path: str = '.[extra]', *, should_raise: bool = False):
    """
    :param name: module name failed to load
    :param should_raise: bool indicating if ImportError should be raised
    """
    msg = f'"{name}" is not installed, use "pip install {default_install_path}" to fulfil requirement'
    if should_raise:
        raise ImportError(msg)
    else:
        default_log(prefix='Requirements').debug(f'{msg} or ignore this warning')
