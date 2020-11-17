from typing import Callable


class SequenceIterator(object):
    """
    The value of this iterator changes according to specified sequence. Iterator is used in parameter-free evolutionary
    scheme for population size control.
    :param sequence_func: the function for sequence generation
    :param start_value: start value of sequence (if start_value doesn't match any value in the sequence, then iterator
    find closest value in sequence)
    :param max_sequence_value: maximal value in sequence
    :param min_sequence_value: minimal value in sequence
    """

    def __init__(self, sequence_func: Callable, start_value: int = None, max_sequence_value: int = None,
                 min_sequence_value: int = None):
        self.sequence_func = sequence_func
        self.archive = {}
        self.start_value = start_value
        self.index = self.get_sequence_index(self.start_value) - 1 if start_value is not None else - 1
        self.max_sequence_value = max_sequence_value
        self.min_sequence_value = min_sequence_value

    def has_prev(self) -> bool:
        if self.index > 0:
            if self.min_sequence_value is not None:
                has = self.sequence_item_calculation(self.index - 1) >= self.min_sequence_value
            else:
                has = True
        else:
            has = False
        return has

    def has_next(self):
        if self.max_sequence_value:
            has = self.max_sequence_value >= self.sequence_item_calculation(self.index + 1)
        else:
            has = True
        return has

    def sequence_item_calculation(self, index: int = None):
        index = self.index if index is None else index
        if index not in list(self.archive):
            result = self.sequence_func(index)
            self.archive[index] = result
        else:
            result = self.archive[index]
        return result

    def get_sequence_index(self, value: int) -> int:
        number = 0
        sequence_value = self.sequence_item_calculation(number)
        while sequence_value < value:
            number += 1
            sequence_value = self.sequence_item_calculation(number)
        return number

    def next(self):
        try:
            self.index += 1
            if self.min_sequence_value is not None:
                if self.sequence_item_calculation(self.index) < self.min_sequence_value:
                    self.index = self.get_sequence_index(self.min_sequence_value)
            result = self.sequence_item_calculation()
        except IndexError:
            raise StopIteration()
        return result

    def prev(self):
        self.index -= 1
        if self.index < 0:
            raise StopIteration()
        return self.sequence_item_calculation()

    def __iter__(self):
        return self


def fibonacci_sequence(n: int) -> int:
    a = 0
    b = 1
    for __ in range(n):
        a, b = b, a + b
    return a
