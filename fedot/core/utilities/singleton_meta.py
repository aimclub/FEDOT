import os

from contextlib import contextmanager
from threading import RLock


class SingletonMeta(type):
    """
    This meta class can provide other classes with the Singleton pattern.
    It guarantees to create one and only class instance.
    Pass it to the metaclass parameter when defining your class as follows:

    class YourClassName(metaclass=SingletonMeta)
    """
    _instances = {}
    _mp_name: str = '_FEDOT_SINGLETON_INSTANCES'
    _lock: RLock = RLock()  # TODO: seems like it's useless in multiprocessing, but that's even from threading lib?!

    @staticmethod
    @contextmanager
    def using_mp():
        # with pathlib.Path(default_fedot_data_dir(), 'dumped').open(mode='w+') as file:
        #   saved_instance = json.dumps(SingletonMeta._instances, cls=Serializer)
        # os.environ[SingletonMeta._mp_name] = saved_instance
        yield
        del os.environ[SingletonMeta._mp_name]

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                # mp_enabled = os.getenv(SingletonMeta._mp_name)
                # if mp_enabled:
                #     from fedot.core.serializers import Serializer
                #     with pathlib.Path(default_fedot_data_dir(), 'dumped').open(mode='r') as file:
                #         cls._instances = json.load(file, cls=Serializer)
                # else:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]
