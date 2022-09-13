import collections.abc

from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Callable, Container, Generic, Iterable, Iterator, List, Optional, Sequence, Sized, TypeVar, Union

T = TypeVar('T')


class UniqueList(list, Generic[T]):
    """
    Simple list that maintains uniqueness of elements.
    In comparison to 'set': preserves list interface and element ordering.

    But this class is better to use only for short lists because of linear complexity of uniqueness lookup.
    Behaves optimally for <= 4 items, good enough for <= 20 items.

    List addition and multiplication are not overloaded and return a standard list.
    But addition with assignment (+=) behaves as UniqueList.extend.

    :param iterable: list of elements to turn into distinct
    """

    def __init__(self, iterable: Optional[Iterable[T]] = None):
        iterable = iterable or ()
        super().__init__(dict.fromkeys(iterable).keys())  # preserves order and uniqueness in the newly created list

    def append(self, value: T):
        """
        Adds ``value`` to the end of the list in case it is not present there

        :param value: will be added to the list or ignored if it's already there
        """
        if value not in super().__iter__():
            super().append(value)

    def extend(self, iterable: Iterable[T]):
        """
        Extends current list with the ``iterable`` elements that aren't present in the list yet

        :param iterable: sequence of elements to be added to the list in case they aren't there
        """
        impl = super()
        impl.extend(element for element in iterable
                    if element not in impl.__iter__())

    def insert(self, index: int, value: T):
        """
        Inserts specified ``value`` to the provided ``index`` in the list in case it's unique value

        :param index: position for inserting into the list
        :param value: will be inserted at the specified ``index`` if it's unique
        """
        if value not in super().__iter__():
            super().insert(index, value)

    def __setitem__(self, key: Union[int, slice], value: Union[T, Iterable[T]]):
        """
        Sets specified ``value`` at the specified index or slice if it's unique value

        :param key: for pointing to the elements from the list
        :param value: element(s) to be set by the specified ``key``
        """
        if value not in super().__iter__():
            super().__setitem__(key, value)

    def __iadd__(self, other: Iterable[T]) -> 'UniqueList':
        """
        Extends current list with the ``iterable`` elements that aren't present in the list yet

        :param other: sequence of elements to be added to the list in case they aren't there

        :return: this class instance
        """
        self.extend(other)
        return self


def remove_items(collection: List[T], items_to_remove: Container[T]):
    """
    Removes all specified items from the list. Modifies original collection

    :param collection: list of elements to be filtered
    :param items_to_remove: filter list of unwanted elements

    :return: modified ``collection`` parameter
    """
    if collection:
        collection[:] = [item for item in collection if item not in items_to_remove]
    return collection


def are_same_length(collections: Iterable[Sized]) -> bool:
    """
    Checks if all arguments have the same length

    :param collections: collection of collections

    :return: does collections inside ``collections`` have the same length
    """
    it = collections.__iter__()
    first = next(it, None)
    if first is not None:
        first = len(first)
        for elem in it:
            if len(elem) != first:
                return False
    return True


def ensure_wrapped_in_sequence(
        obj: Optional[Union[T, Iterable[T]]],
        sequence_factory: Callable[[Iterable[T]], Sequence[T]] = list
) -> Optional[Sequence[T]]:
    """
    Makes sure given ``obj`` is of type that acts like sequence and converts ``obj`` to it otherwise

    :param obj: any object to be ensured or wrapped into sequence type
    :param sequence_factory: sequence factory for wrapping ``obj`` into in case it is not of any sequence type

    :return: the same object if it's of type sequence (or None)
        or ``obj`` wrapped in type provided by ``sequence_factory``
    """
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

    def __eq__(self, other: 'Enum'):
        """
        Compares this enum with the ``other_graph``

        :param other: another enum

        :return: is it equal to ``other`` in terms of the string representation
        """
        return str(self) == str(other)

    def __hash__(self):
        """
        Gets hashcode of this enum

        :return: hashcode of string representation of this enum
        """
        return hash(str(self))


class Copyable:
    """Provides default implementations for `copy` & `deepcopy`."""

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo=None):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result


class Comparable(ABC):
    @abstractmethod
    def __eq__(self, other: 'Comparable') -> bool:
        """
        Compares this object of :class:`Comparable` interface with the ``other`` comparable

        :param other: other object implementing :class:`Comparable` interface

        :return: is it equal to ``other`` in terms of the comparables
        """
        raise NotImplementedError()

    @abstractmethod
    def __lt__(self, other: 'Comparable') -> bool:
        """
        Compares this object of :class:`Comparable` interface with the ``other`` comparable

        :param other: other object implementing :class:`Comparable` interface

        :return: is less than ``other`` in terms of the comparables
        """
        raise NotImplementedError()

    def __ne__(self, other: 'Comparable') -> bool:
        """
        Compares this object of :class:`Comparable` interface with the ``other`` comparable

        :param other: other object implementing :class:`Comparable` interface

        :return: is it not equal to ``other`` in terms of the comparables
        """
        return not self.__eq__(other)

    def __le__(self, other) -> bool:
        """
        Compares this object of :class:`Comparable` interface with the ``other`` comparable

        :param other: other object implementing :class:`Comparable` interface

        :return: is less than or equal to ``other`` in terms of the comparables
        """
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other) -> bool:
        """
        Compares this object of :class:`Comparable` interface with the ``other`` comparable

        :param other: other object implementing :class:`Comparable` interface

        :return: is greater than ``other`` in terms of the comparables
        """
        return not self.__le__(other)

    def __ge__(self, other) -> bool:
        """
        Compares this object of :class:`Comparable` interface with the ``other`` comparable

        :param other: other object implementing :class:`Comparable` interface

        :return: is greater than or equal to ``other`` in terms of the comparables
        """
        return not self.__lt__(other)


class BidirectionalIterator(Iterator[T]):
    @abstractmethod
    def has_prev(self) -> bool:
        """
        Checks if this iterator has implemented previous item getter

        :return: whether this iterator has previous item getter
        """
        raise NotImplementedError()

    @abstractmethod
    def has_next(self) -> bool:
        """
        Checks if this iterator has implemented next item getter

        :return: whether this iterator has next item getter
        """
        raise NotImplementedError()

    @abstractmethod
    def next(self) -> T:
        """
        Gets next item of this iterator

        :return: next item
        """
        raise NotImplementedError()

    @abstractmethod
    def prev(self) -> T:
        """
        Gets previous item of this iterator

        :return: previous item
        """
        raise NotImplementedError()

    def __next__(self):
        """
        Gets next item of this iterator

        :raises StopIteration: the end of this iterator

        :return: next item
        """
        if self.has_next():
            return self.next()
        raise StopIteration()

    def __iter__(self):
        """
        Returns this iterator

        :return: this iterator
        """
        return self
