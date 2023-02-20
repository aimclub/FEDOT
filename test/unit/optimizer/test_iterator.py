from golem.core.utilities.sequence_iterator import SequenceIterator, fibonacci_sequence


def test_iterator_without_constraints():
    sequence_generation_function = fibonacci_sequence
    iterator = SequenceIterator(sequence_func=sequence_generation_function)
    num_of_values = 5
    sequence_values = [iterator.next() for _ in range(num_of_values)]
    assert sequence_values[0] == 0 and sequence_values[4] == 3


def test_iterator_from_certain_index():
    sequence_generation_function = fibonacci_sequence
    iterator = SequenceIterator(sequence_func=sequence_generation_function, start_value=3)
    num_of_values = 5
    sequence_values = [iterator.next() for _ in range(num_of_values)]
    assert sequence_values[0] == 3 and sequence_values[4] == 21


def test_iterator_with_max_min_constraints():
    sequence_generation_function = fibonacci_sequence
    iterator = SequenceIterator(sequence_func=sequence_generation_function, min_sequence_value=10,
                                max_sequence_value=25)
    assert not iterator.has_prev()
    first_value = iterator.next()
    second_value = iterator.next()
    assert first_value == 13
    assert second_value == 21
    assert not iterator.has_next()
    assert iterator.has_prev()
