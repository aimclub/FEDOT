import os
import pickle

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
        pickled = str(pickle.dumps(SingletonMeta._instances, 0), 'ascii')
        os.environ[SingletonMeta._mp_name] = pickled
        yield
        del os.environ[SingletonMeta._mp_name]

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                saved_instances = os.getenv(SingletonMeta._mp_name)
                if saved_instances:
                    cls._instances = pickle.loads(bytes(saved_instances, 'ascii'))
                else:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]
