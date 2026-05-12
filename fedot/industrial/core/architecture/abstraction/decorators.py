from contextlib import contextmanager
from typing import Optional, Callable
from weakref import WeakValueDictionary

from distributed import Client, LocalCluster
from fedot.core.data.input_data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot.industrial.core.architecture.preprocessing.data_convertor import CustomDatasetCLF, CustomDatasetTS, DataConverter, \
    TensorConverter
from fedot.industrial.core.architecture.settings.computational import backend_methods as np


def fedot_data_type(func):
    def decorated_func(self, *args):
        if not isinstance(args[0], InputData):
            args[0] = DataConverter(data=args[0])
        features = args[0].features

        if len(features.shape) < 4:
            try:
                input_data_squeezed = np.squeeze(features, 3)
            except ValueError:
                input_data_squeezed = np.squeeze(features)
        else:
            input_data_squeezed = features
        return func(self, input_data_squeezed, args[1])

    return decorated_func


def convert_to_4d_torch_array(func):
    def decorated_func(self, *args):
        init_data = args[0]
        if isinstance(init_data, tuple):
            return func(self, init_data)
        else:
            data = DataConverter(data=init_data).convert_to_4d_torch_format()
            if isinstance(init_data, InputData):
                init_data.features = data
            else:
                init_data = data
            return func(self, init_data)

    return decorated_func


def convert_to_3d_torch_array(func):
    def decorated_func(self, *args):
        init_data = args[0]
        data = DataConverter(data=init_data).convert_to_torch_format()
        if isinstance(init_data, InputData):
            init_data.features = data
        else:
            init_data = data
        return func(self, init_data, *args[1:])

    return decorated_func


def convert_inputdata_to_torch_dataset(func):
    def decorated_func(self, *args):
        ts = args[0]
        return func(self, CustomDatasetCLF(ts))

    return decorated_func


def convert_inputdata_to_torch_time_series_dataset(func):
    def decorated_func(self, *args):
        ts = args[0]
        return func(self, CustomDatasetTS(ts))

    return decorated_func


def convert_to_torch_tensor(func):
    def decorated_func(self, *args):
        data = TensorConverter(data=args[0]).convert_to_tensor(data=args[0])
        return func(self, data)

    return decorated_func


def remove_1_dim_axis(func):
    def decorated_func(self, *args):
        time_series = np.nan_to_num(args[0])
        if any([dim == 1 for dim in time_series.shape]):
            time_series = DataConverter(data=time_series).convert_to_1d_array()
        return func(self, time_series)

    return decorated_func


def convert_to_input_data(func):
    def decorated_func(*args, **kwargs):
        features, names = func(*args, **kwargs)
        ts_data = InputData(idx=np.arange(len(features)),
                            features=features,
                            target='no_target',
                            task='no_task',
                            data_type=DataTypesEnum.table,
                            supplementary_data={'feature_name': names})
        return ts_data

    return decorated_func


class Singleton(type):
    _instances = WeakValueDictionary()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super(Singleton, cls).__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class DaskServer(metaclass=Singleton):
    def __init__(self, params: Optional[OperationParameters] = None):
        print('Creating Dask Server')
        cluster_params = params.get('cluster_params', dict(processes=False,
                                                           n_workers=1,
                                                           threads_per_worker=4,
                                                           memory_limit='auto'
                                                           ))
        cluster = LocalCluster(**cluster_params)
        # connect client to your cluster
        self.client = Client(cluster)
        self.cluster = cluster


@contextmanager
def exception_handler(*exception_types, on_exception: Optional[Callable] = None, suppress: bool = True):
    """
    A context manager that wraps code with a try-except block.

    Args:
        *exception_types (tuple): The types of exceptions to catch (e.g., ValueError, TypeError).
        on_exception (callable, optional): A function to call when an exception is caught.
        suppress (bool, optional): If True, suppresses the exception after handling it.
            If False, re-raises the exception after handling it. Defaults to True.

    Returns:
        None: This context manager does not return any value directly.
              However, it controls the flow of execution within the `with` block.
    """
    try:
        yield  # Executes the code within the 'with' block
    except exception_types:
        if on_exception:
            on_exception()  # Call the provided callback function
        if not suppress:
            raise  # Re-raise the exception if suppression is disabled
