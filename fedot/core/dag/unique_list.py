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
