from abc import ABC, abstractmethod

from typing import Tuple, Union, Optional


class Serializable(ABC):

    @classmethod
    def from_serialized(cls, source: Union[str, dict], internal_state_data: Optional[dict] = None):
        """
        Static constructor for convenience. Creates default instance and calls .load() on it.
        """
        default_instance = cls()
        default_instance.load(source, internal_state_data)
        return default_instance

    @abstractmethod
    def save(self, path: str = None, datetime_in_path: bool = True) -> Tuple[str, dict]:
        """
        Save the graph to a json representation
        with pickled custom data (e.g. fitted models).

        :param path: path to json file with operation
        :param datetime_in_path: flag for addition of the datetime stamp to saving path
        :return: json containing a composite operation description
        """
        raise NotImplementedError()

    @abstractmethod
    def load(self, source: Union[str, dict], internal_state_data: Optional[dict] = None):
        """
        Load the graph from json representation, optionally
        with pickled custom internal data (e.g. fitted models).

        :param source: path to json file with operation or json dictionary itself
        :param internal_state_data: dictionary of the internal state
        """
        raise NotImplementedError()
