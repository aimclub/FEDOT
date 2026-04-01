""" This file is part of the FEDOT framework for automated machine learning. """

from fedot.version import __version__

try:
    from fedot.api import Fedot, FedotBuilder
except ModuleNotFoundError as ex:
    if ex.name and not ex.name.startswith('golem'):
        raise
    Fedot = None
    FedotBuilder = None

__all__ = ['Fedot', 'FedotBuilder', '__version__']
