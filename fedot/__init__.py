""" This file is part of the FEDOT framework for automated machine learning. """

from fedot.api import Fedot, FedotBuilder, create_data, create_data_lazy
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.validation import FedotInvalidKeysError, FedotValidationError, ValidationContext
from fedot.version import __version__
