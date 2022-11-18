from golem.core.utilities.data_structures import UniqueList


def test_init():
    assert UniqueList() == []
    assert UniqueList(None) == []
    assert UniqueList(range(5)) == list(range(5))
    assert UniqueList(list(range(5))) == list(range(5))


def test_list_arithmetic():
    assert UniqueList(range(0, 5)) + UniqueList(range(3, 8)) == list(range(5)) + list(range(3, 8))
    assert UniqueList(range(0, 5)) + list(range(3, 8)) == list(range(5)) + list(range(3, 8))
    assert UniqueList(range(3)) * 3 == list(range(3)) * 3


def test_append_same_element_twice():
    xs = UniqueList(range(5))
    assert len(xs) == 5

    xs.append(5)
    assert xs.count(5) == 1
    assert len(xs) == 6
    # same element is not added second time
    xs.append(5)
    assert len(xs) == 6
    assert xs.count(5) == 1


def test_extend():
    xs = UniqueList(range(5))
    # Only unique elements are added
    xs.extend([5, 6, 7, 7, 4, 8, 9, 0, 9])
    assert xs == list(range(10))

    xs = UniqueList(range(5))
    # Only unique elements are added
    xs += [5, 6, 7, 7, 4, 8, 9, 0, 9]
    assert xs == list(range(10))


def test_insert():
    xs = UniqueList(range(10))

    len_previous = len(xs)
    xs.insert(2, 5)
    assert len(xs) == len_previous
    xs.insert(2, 11)
    assert len(xs) == len_previous + 1

    from copy import copy

    previous = copy(xs)
    xs[3] = 5
    assert len(xs) == len(previous)
    assert xs[3] == previous[3]
    xs[3] = 12
    assert len(xs) == len(previous)
    assert xs[3] == 12
