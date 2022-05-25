import collections.abc

from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Container, Iterator, Iterable, List, Sequence, Sized, TypeVar, Union, Optional


class UniqueList(list):
    """ Simple list that maintains uniqueness of elements.
    In comparison to 'set': preserves list interface and element ordering.

    But this class is better to use only for short lists
    because of linear complexity of uniqueness lookup.
    Behaves optimally for <= 4 items, good enough for <= 20 items.

    List addition and multiplication are not overloaded and return a standard list.
    But addition with assignment (+=) behaves as UniqueList.extend.
    """

    def __init__(self, iterable=None):
        seen = set()
        iterable = iterable or ()
        super().__init__(seen.add(element) or element
                         for element in iterable
                         if element not in seen)
        del seen

    def append(self, value):
        if value not in super().__iter__():
            super().append(value)

    def extend(self, iterable):
        impl = super()
        impl.extend(element for element in iterable
                    if element not in impl.__iter__())

    def insert(self, index, value):
        if value not in super().__iter__():
            super().insert(index, value)

    def __setitem__(self, key, value):
        if value not in super().__iter__():
            super().__setitem__(key, value)

    def __iadd__(self, other):
        self.extend(other)
        return self


def remove_items(collection: List, items_to_remove: Container):
    """Removes all specified items from the list. Modifies original collection."""
    if collection:
        collection[:] = [item for item in collection if item not in items_to_remove]
    return collection


def are_same_length(collections: Iterable[Sized]) -> bool:
    """Checks if all arguments have the same length."""
    it = collections.__iter__()
    first = next(it, None)
    if first is not None:
        first = len(first)
        for elem in it:
            if len(elem) != first:
                return False
    return True


T = TypeVar('T')


def ensure_wrapped_in_sequence(
        obj: Optional[Union[T, Iterable[T]]],
        sequence_factory: Callable[[Iterable[T]], Sequence[T]] = list
) -> Optional[Sequence[T]]:
    if obj is None:
        return obj
    elif isinstance(obj, str) or not isinstance(obj, collections.abc.Iterable):
        return sequence_factory([obj])
    elif not isinstance(obj, collections.abc.Sequence) or not isinstance(obj, sequence_factory):
        return sequence_factory(obj)
    return obj


class ComparableEnum(Enum):
    """
    The Enum implementation that allows to avoid the multi-module enum comparison problem
    (https://stackoverflow.com/questions/26589805/python-enums-across-modules)
    """

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))


class Comparable(ABC):
    @abstractmethod
    def __eq__(self, other) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __lt__(self, other) -> bool:
        raise NotImplementedError()

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __le__(self, other) -> bool:
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other) -> bool:
        return not self.__le__(other)

    def __ge__(self, other) -> bool:
        return not self.__lt__(other)


class BidirectionalIterator(Iterator[T]):
    @abstractmethod
    def has_prev(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def has_next(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def next(self) -> T:
        raise NotImplementedError()

    @abstractmethod
    def prev(self) -> T:
        raise NotImplementedError()

    def __next__(self):
        if self.has_next():
            return self.next()
        raise StopIteration()

    def __iter__(self):
        return self
